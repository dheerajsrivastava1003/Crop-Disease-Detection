# Crop-Disease-Detection
Developed a deep learning-based crop disease detection system using ResNet and DenseNet architectures to identify plant diseases from leaf images. Implemented image preprocessing, data augmentation, and model evaluation using Python, PyTorch, and OpenCV to improve prediction accuracy and support early disease detection in agriculture.
Features
Detects crop diseases from uploaded leaf images
Deep learning-based image classification
Hybrid ResNet + DenseNet architecture
User-friendly prediction interface
High accuracy disease detection
Real-time prediction support
Technologies Used
Python
PyTorch
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
Streamlit / Flask (if used for deployment)
Dataset

The dataset consists of labeled crop leaf images containing healthy and diseased plant samples. Images were preprocessed using resizing, normalization, and augmentation techniques to improve model performance.

Dataset Source: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Model Architecture

The project uses a Hybrid CNN Model combining:

ResNet for deep feature extraction
DenseNet for feature reuse and improved gradient flow
Fully connected classifier head for final disease prediction

Additional techniques used:

Transfer Learning
Data Augmentation
Dropout Regularization
Early Stopping
Learning Rate Scheduling
Workflow
Dataset Collection
Data Preprocessing
Data Augmentation
Model Training
Model Evaluation
Disease Prediction
Deployment
