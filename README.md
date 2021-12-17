# How to use
The code contains both binary classification and multi-classification implementations of brain tumour images.
The binary classification task is to build a classifier to identify whether there is a tumor in the MRI images.  
The multi-classification task is to build a classifier to identify the type of tumor in each MRI image (meningioma tumor, glioma tumor, pituitary tumor or no tumor). 

# Binary Classification Task 
The code is mainly written in jupyter-notebook.  
Just open jupyter-notebook to run these four codes directly, and each specific function can be clearly seen in the file.  

The following documents are included:
1. CNN-Classificaiton uses a simple neural network for binary classification.    
2. KNN-Classificaiton uses the K-nearest neighbor algorithm for binary classification.    
3. Logistic-Regression uses the logistic regression algorithm for binary classification.    
4. SVM-Classificaiton uses the SVM for binary classification.  
5. The image file contains 3000 512x512 pixel gray-scale MRI images of the brain tumour organized in 4 classes.   
6. The lable.csv file contains the categories corresponding to each brain tumour image.

# Multi-Classificaiton Task
The code is mainly written in pycharm, the .py file can be opened and executed with pycharm.  
In particular, the predict_one_result file needs to be opened and run with jupyter-notebook.   
Before running it, you need to run a neural network file to get and save its model, the default is Xception, so you need to run the Xception_keras.py file first and then run predict_one_result, in which you can change the model and the image you want to use.

The following documents are included:  
1. Alex_binary_classification.py uses AlexNet to complete a binary classification task.  
2. AlexNet_multiclassification.py using AlexNet for a multi-classification task.  
3. ResNet.py using ResNet50 for a multi-classification task.    
4. ResNet_model.py contains three models, ResNet34, ResNet50 and ResNet101, which can be called according to your needs.  
5. ResNet_transfer_learning.py calls the ResNet50 model in ResNet_model.py and uses the pre-training parameters provided by the pretrain_weights file to perform transfer            learning to complete the multi-classification task.   
6. Xception.py uses the Xception model for a multi-classification task  
7. Xception_keras.py calls the Xception model provided by Keras library for the multi-classification task.  
8. The split.py and Tumor_images_split files in the Data file are designed to put all the images in the image path into four folders according to their respective classes, this    step is already done and the user does not need to run these two codes again.  
   The train folder contains the training set images divided by the four tumour classes, and the val folder contains the validation set images.
   The test folder contains 200 images of brain tumours for testing.  
9. The output folder is used to store the saved models and the output curves, which already contain some plotted curves.  
10. The pretrain_weights folder holds the downloaded pre-training parameter files for transfer learning.  
11. The save_weights folder is used to store the weight parameters in the training.  
12. The test_label file holds the tumour categories for each of the 200 test set images.  
13. predict_one_result file predicts the category of the image using the saved model.  
14. The class_indices file holds the four tumour classes.

# Library:
Python                   3.8  
matplotlib               3.4.2    
numpy                    1.19.5    
pandas                   1.3.3  
scikit-learn             0.24.2  
scikit-image             0.18.3  
scipy                    1.4.1  
tensorflow-gpu           2.5.0  
Cuda                     11.2  
Pillow                   8.4.0  
tqdm                     4.62.3
