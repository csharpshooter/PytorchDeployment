import io
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
from PIL import Image
import torchvision.models as models
from app.resnet import ResNet18

num_classes = 10

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

PATH = "app/model.pth"
model = ResNet18().to('cpu')
model.load_state_dict(torch.load(PATH,map_location ='cpu'))
model.eval()

def transform_image(image_bytes):

    transform = transforms.Compose([transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)
    
    
    
def get_prediction(image_tensor):
# move the input and model to GPU for speed if available
    input_batch = image_tensor
    if torch.cuda.is_available():
        input_batch = image_tensor.to('cuda')
        model.to('cuda')   

    with torch.no_grad():
        output = model(input_batch)     
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)
    _, predictions = torch.max(output, 1)
    # predicted_idx = predictions.item()
    #print(predictions.item())
    #print(predictions.item())
    class_id = predictions.item()
    class_name = str(classes[predictions.item()])
    #result = {'predicted class id': predictions.item(), 'predicted class': str(classes[predictions.item()])}
    return class_id, class_name
