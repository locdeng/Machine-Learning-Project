# Machine-Learning-Project
The Practice of Supervised Learning and Semi-Supervised Learning

Using Google Colab to do this project.


Appying the given reference models (VGG, ResNet, MobileNet) to implement the project.


Datasets:

 Task 1: Supervised Learning
- Image dataset of 96×96 size (number of classes: 50).
- Using 30,000 images (each class has 600 images) as a labeled training set 
and 2500 for a validation set.

 Task 2: Semi-Supervised Learning
- Image dataset of 96×96 size (number of classes: 10).
- Using 6,000 images (each class has 600 images) as labeled training datasets, 
6,000 images as unlabeled training sets, and 500 for validation sets.

Training Process:

 Task 1: Supervised Learning
- To train a deep learning model to classify images into 50 labels using 
the provided image datasets with annotated labels.

Task 2: Semi-Supervised Learning
- To train the deep learning models to classify images into 10 labels
using labeled and unlabeled images.
- To use the co-training method to learn two different CNN models
simultaneously to utilize unlabeled images to boost the classification performance.
- Take two CNN models to perform co-training for semi-supervised learning.
- Two models trained by labeled dataset take in the unlabeled set and
compare the results. The results of two models that match together (if they classify 
the same label) produce pseudo labels that can be utilized as training sets.


