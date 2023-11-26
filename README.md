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

# General steps for fine-tuning a pre-trained CNN model:
1. Label images:
- For image classification tasks, the labeling process involves assigning a class label to each image and organizing them into folders based on their classes. Tools like RoboFlow or CVAT simplify this process.


2. Split Data:
- Divide dataset into training and validation sets. The training set is used to train the model, while the validation set is used to evaluate its performance during training.


3. Pre-process the images: (Resize images and batching dataset)
- Resize images into appropriate sizes, each pre-trained model has a different image size. For EfficientNet, the image size used was 128x128 for the smallest model.
- Batch size for image classification can vary, a good starting point is 32 for batch size.


4. Augment Images (if needed):
- Data augmentation techniques can be applied to artificially increase the diversity
of your dataset. This helps the model generalize better.


5. Train the model:
- The models were trained using PyTorch, feel free to use tensorflow if you want. Monitor metrics such as train loss and val loss during training.


6. Validate the Model:
- Evaluate the trained model on the validation set to ensure it generalizes well
to new, unseen data.


7. Inference on Test Set:
- After training, perform inference on a separate test set to assess the model's performance on completely unseen data.







