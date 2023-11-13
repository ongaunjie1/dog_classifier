# Dog Classifier App (trained using PyTorch)

# Important Note
* Pre-trained model used was EfficientNet_B0 (model pre-trained on ImageNet dataset) : https://github.com/lukemelas/EfficientNet-PyTorch
* Custom model was trained using the smallest EfficientNet, model imported using torch image model (timm) : https://huggingface.co/timm
* Model Trained on 120 different breeds of dog, refer to dog.txt for the list of dog breeds trained
* For the purpose of using Streamlit cloud, the smallest model is chosen for a faster inference speed, with less accuracy
* Accuracy of the model is **75%** which is expected for using the smaller model

# Further improvements for improving model performance
* Acquire higher quality images
* Perform Data Augmentation,
* Perform Hyperparameter tuning
* Experiment with EfficientNetV2
* Experiment with different pre-trained image classifier models such as ResNET

# Link to app
* https://dogclassifier-efficientnet.streamlit.app/

# How to use the app?
## Step 1: Upload the image of a dog
![image](https://github.com/ongaunjie1/dog_classifier/assets/118142884/1dcab942-cf50-4a13-9100-10817fad4541)

## Step 2: Click on the classify button
![image](https://github.com/ongaunjie1/dog_classifier/assets/118142884/24d497da-b462-4993-9f51-0d1a94f07184)








