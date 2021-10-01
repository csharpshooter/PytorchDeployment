# Pytorch deployment - Image classification of CIFAR-10 dataset 

## Description
Deploy our PyTorch model with Flask and Heroku. Create a simple Flask app with a REST API that returns the result as json data,  and then deploy it to Heroku. 
Here we will do image classification, and we can send images to our heroku app and then predict it with our live running app.

## Heroku App Link :
https://pytorch-deploy-test.herokuapp.com/predict

#### App Demo deployed on Heroku :
![Demo of deployed app](https://github.com/csharpshooter/PytorchDeployment/blob/main/pytorch-deploy-test-small.gif)

#### Trained model using Kuang-Liu Resnet-18 model on Cifar-10 dataset from scratch for 20 epochs. 
#### Train Accuracy: 87.352 Test Accuracy: 86.9
1. Training code, model output and metrics.csv is located in .PytorchDeployment/train. Initialized repo with dvc and tracked changes using dvc.
2. Achieved more than 70% accuracy on all classes:     
    Accuracy on the test images: 85 %   
    Accuracy for class airplane is: 87.7 %   
    Accuracy for class automobile is: 93.4 %    
    Accuracy for class bird  is: 78.5 %   
    Accuracy for class cat   is: 75.2 %   
    Accuracy for class deer  is: 83.3 %   
    Accuracy for class dog   is: 74.1 %   
    Accuracy for class frog  is: 89.2 %   
    Accuracy for class horse is: 92.5 %   
    Accuracy for class ship  is: 90.1 %   
    Accuracy for class truck is: 92.3 %   
3. Wrote following 6 test cases in unittest.py. Used pytest for writing unit tests.   
    test_check_if_model_file_present_in_root_folder   
    test_check_if_data_folder_present_in_root_folder    
    test_check_if_metrics_csv_present_in_root_folder    
    test_validate_train_accuracy_greater_than_70_pct    
    test_validate_test_accuracy_greater_than_70_pct   
    test_validate_individual_class_accuracy_greater_than_70_pct   

#### Current Status of repository ####
[![Python application test](https://github.com/csharpshooter/PytorchDeployment/actions/workflows/python-app-tests.yml/badge.svg)](https://github.com/csharpshooter/PytorchDeployment/actions/workflows/python-app-tests.yml)
