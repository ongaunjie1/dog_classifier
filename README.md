# Dog Classifier App

# Important Note
* Pre-trained model used was EfficientNet_B0 (model pre-trained on ImageNet dataset) : https://github.com/lukemelas/EfficientNet-PyTorch
* Custom model was trained using the smallest EfficientNet, model imported using torch image model (timm) : https://huggingface.co/timm
* Model Trained on 120 different breeds of dog, refer to dog.txt for the list of dog breeds trained
* For the purpose of using Streamlit cloud, the smallest model is chosen for a faster inference speed, with less accuracy
* Accuracy of the model is **0.75%** which is expected for the smaller model

# Further improvements for improving model performance
* Acquire higher quality images
* Perform Data Augmentation,
* Perform Hyperparameter tuning
* Experiment with EfficientNetV2
* Experiment with different pre-trained image classifier models such as ResNET

# Link to app
* https://dogclassifier-efficientnet.streamlit.app/https://dogclassifier-efficientnet.streamlit.app/






