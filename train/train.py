import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from csv_logger import CsvLogger
import logging
import sys

print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
from subprocess import call
# call(["nvcc", "--version"]) does not work
#nvcc --version
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('Active CUDA Device: GPU', torch.cuda.current_device())

print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

filename = 'metrics.csv'
level = logging.INFO
fmt = '%(message)s'
datefmt = '%Y/%m/%d %H:%M:%S'
max_size = 1024  # 1 kilobyte
max_files = 4  # 4 rotating files
header = ['epoch', 'accuracy', 'loss', 'val_accuracy', 'val_loss']

# Creat logger with csv rotating handler
csvlogger = CsvLogger(
    filename=filename,
    level=level,
    fmt=fmt,
    datefmt=datefmt,
    max_size=max_size,
    max_files=max_files,
    header=header
)

print(torch.__version__)

batch_size = 64

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
    
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

    
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)



def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

dataiter = iter(trainloader)
images, labels = dataiter.next()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = models.resnet18(pretrained=True).to(device)

criterion = nn.CrossEntropyLoss()

train_losses = []
test_losses = []
train_acc = []
test_acc = []
t_acc_max = 0
reg_loss_l1 = []
factor = 0  # 0.00005
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
PATH = 'model.pth'
loss_type = nn.CrossEntropyLoss()


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    loss_batch = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate L1 loss
        l1_crit = torch.nn.L1Loss(size_average=False)
        reg_loss = 0
        for param in model.parameters():
            spare_matrix = torch.randn_like(param) * 0
            reg_loss += l1_crit(param, spare_matrix)

        reg_loss_l1.append(reg_loss)

        # Calculate loss
        loss = loss_type(y_pred, target)
        loss_batch += loss.item()
        loss += factor * reg_loss


        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

    train_acc.append(100 * correct / processed)
    train_losses.append(loss_batch / len(train_loader.dataset))



def test(model, device, test_loader, class_correct, class_total, epoch, t_acc_max):
    model.eval()
    test_loss = 0
    correct = 0
    t_acc = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_type(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct_new = np.squeeze(correct_tensor.cpu().numpy())

            # calculate test accuracy for each object class
            for i in range(10):
                label = target.data[i]
                class_correct[label] += correct_new[i].item()
                class_total[label] += 1

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))
    t_acc = 100. * correct / len(test_loader.dataset)

    # save model if validation loss has decreased
    if t_acc_max <= t_acc:
        print('Validation accuracy increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            t_acc_max,
            t_acc))
        t_acc_max = t_acc
        torch.save(net.state_dict(), PATH)

    return t_acc, t_acc_max, test_loss


optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
epochs = 10

for epoch in range(epochs):
    print("EPOCH:", epoch)
    train(net, device, trainloader, optimizer, epoch)
    t_acc, t_acc_max,test_loss = test(net, device, testloader, class_correct, class_total, epoch, t_acc_max)

for epoch in range(epochs):
    csvlogger.info([epoch + 1, train_acc[epoch], train_losses[epoch], test_acc[epoch], test_losses[epoch]])

print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

print('Loading Saved Model')

# net = models.resnet18(pretrained=False).to(device)
# net.load_state_dict(torch.load(PATH))

print('Loading Completed')
images = images.to(device)
labels = labels.to(device)
outputs = net(images)

# acc = [0 for c in classes]
# for c in classes:
#     acc[c] = ((preds == labels) * (labels == c)).float() / (max(labels == c).sum(), 1))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on the test images: %d %%' % (
        100 * correct / total))

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

csvlogger.info([])
csvlogger.info(['Train Accuracy: {}'.format(max(train_acc))])
csvlogger.info(['Test Accuracy: {}'.format(max(test_acc))])
csvlogger.info([])
csvlogger.info(['ClassName','Accuracy'])

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    csvlogger.info([classname,accuracy])
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                         accuracy))