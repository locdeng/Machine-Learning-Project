# Machine-Learning-Types
The Practice of Supervised Learning, Semi-Supervised Learning and Unsupervised Learning

Using Google Colab to do this project.

# Supervised Learning and Semi-Supervised

Applying the given reference models (VGG, ResNet, MobileNet) to implement the project.


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

# Unsupervised-Learning
To Practice Unsupervised Learning Methods in Machine Learning


The experimental setup:
- Using the Google Colab to implement.
- The dataset is Fashion MNIST which is a collection of 10 classes, 60,000 training images 
and 10,000 test images and 28×28 grayscale images.
- Dimension Reduction: PCA is performed on the test images using different dimensions: 784, 
100, 50, and 10, respectively.
- Using dataloader to load the dataset. The image is stored in FM_test_images and the label is 
stored in FM_test_labels.
- Using k-means and DBSCAN algorithms:
 + k-means assigns each data point to one of the clusters based on their proximity.
 + DBSCCAN is applied with 2 hyperparameters: eps and min_neighbors. It identifies 
dense regions in the data and assigns labels to points accordingly.
- ARI calculation
- Visualization
- Generating a table to show the ARI scores

  
The result of the test:

+ For the k-means

![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/07349570-065d-4d18-8de4-79308cc60462)
![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/47cbb4a4-176d-4547-a32c-ba5cebab1991)


![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/4e73b255-9516-4c1d-9072-2de92060432f)
![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/5b8c7180-94ee-4ee4-902e-c78257e63921)


![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/bb5ec2ca-78d8-47a7-9315-b517acbf845f)
![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/93ba7401-fb3b-4a3f-812b-90441bc763a3)


![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/ed41d817-982f-4656-b368-2071bdb683cc)
![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/d190f535-34a7-4696-992d-6529645938b1)



![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/486de033-b78c-4d92-af50-cf4f2c39c3ab)



+ For the DBSCAN:

![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/1cbb589d-64af-4021-983d-da3d908668ce)
![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/eae499ab-3317-42d3-89b8-8b3c950528d0)



![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/aca0e3b2-f040-4601-90cf-83ab554ba714)
![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/53754d8a-d990-40f6-a534-652222769e2b)



![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/05df7405-14ae-474d-b83b-7ee8a67e3418)
![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/6be5be37-6512-4107-8598-98023cce8f66)


![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/d1b50eee-5444-47d2-b514-8d3d77120b5a)
![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/b94bb8b0-749a-4bb8-9a62-495ac6b50809)



![image](https://github.com/locdeng/Unsupervised-Learning/assets/104445003/aae38045-9a6b-40dd-ad70-6346aaa79511)






