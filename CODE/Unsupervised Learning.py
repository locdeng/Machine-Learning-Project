import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms

Fashion_mnist_test_transform = transforms.Compose([transforms.ToTensor()])

testset_Fashion_mnist = datasets.FashionMNIST(root='./data', train=False, download=True,
transform = Fashion_mnist_test_transform)

FM_test = DataLoader(testset_Fashion_mnist, batch_size=32, shuffle=False, num_workers=2)

FM_test_images = []
FM_test_labels = []

for batch in FM_test:
    images, labels = batch
    images_flat = images.view(images.shape[0], -1)
    FM_test_images.append(images_flat.numpy())
    FM_test_labels.append(labels.numpy())
FM_test_images = np.vstack(FM_test_images)
FM_test_labels = np.concatenate(FM_test_labels)

X_ = pd.DataFrame(data=FM_test_images) # test data
y_ = pd.Series(data=FM_test_labels) # test label

pca = PCA(n_components= 50)
test_PCA = pca.fit_transform(X_)
test_PCA = pd.DataFrame(data = test_PCA)

testDF = pd.DataFrame(data=test_PCA.loc[:,0:1], index=test_PCA.index)
testDF = pd.concat((testDF,y_), axis=1, join="inner")
testDF.columns = ["x-axis", "y-axis", "Label"]
sns.lmplot(x="x-axis", y="y-axis", hue="Label", data=testDF, fit_reg=False, height=8)
plt.grid()

n_components = 2
learning_rate = 300
perplexity = 30
early_exaggeration = 12
init = 'random'

tSNE = TSNE(n_components=n_components, learning_rate=learning_rate,
         perplexity=perplexity, early_exaggeration=early_exaggeration, init=init)
tSNE = TSNE(n_components=n_components,)

# Perform PCA with different dimensions
pca_dims = [784, 100, 50, 10]
k = 10 
test_PCA = {}
kmeans = {}
ari_scores = []

# Set DBSCAN hyperparameters
eps = 0.5
min_neighbors = 5

# Perform DBSCAN with different dimensions
dbscan_dims = [784, 100, 50, 10]
test_DBSCAN = {}
ari_scores = []

for dim in pca_dims:
    # Perform PCA
    pca = PCA(n_components=dim)
    test_PCA[dim] = pca.fit_transform(X_)

    
    # Run k-means clustering
    kmeans[dim] = KMeans(n_clusters=k, random_state=42)
    kmeans[dim].fit(test_PCA[dim])
    labels = kmeans[dim].labels_
    


    # Compute ARI
    ari = adjusted_rand_score(y_, labels)
    ari_scores.append(ari)

    # Create a DataFrame for visualization
    testDF = pd.DataFrame(data=test_PCA[dim][:, :2], columns=["x-axis", "y-axis"])
    testDF["Label"] = labels
 
    
    # Plot the results for K-means
    sns.lmplot(x="x-axis", y="y-axis", hue="Label", data=testDF, fit_reg=False, height=8)
    plt.title(f"K-means Clustering Result (PCA Dimension = {dim})")
    plt.grid()
    plt.show()
  

    # Perform t-SNE on the reduced dimensions
    tSNE = TSNE(n_components=2, random_state=42)
    X_tSNE = tSNE.fit_transform(test_PCA[dim])

    # Randomly choose 100 samples for visualization
    random_indices = np.random.choice(X_tSNE.shape[0], size=100, replace=False)
    X_tSNE_sample = X_tSNE[random_indices]
    y_sample = y_[random_indices]

    # Create a DataFrame for visualization
    tSNE_df = pd.DataFrame(data=X_tSNE_sample, columns=["x-axis", "y-axis"])
    tSNE_df["Label"] = y_sample

    # Plot the t-SNE visualization
    sns.lmplot(x="x-axis", y="y-axis", hue="Label", data=tSNE_df, fit_reg=False, height=10)
    plt.title(f"t-SNE Visualization (PCA Dimension = {dim})")
    plt.grid()
    plt.show()

# Create a table for ARI scores
ari_table = pd.DataFrame(data={"Dimension": pca_dims, "ARI Score": ari_scores})

print("Adjusted Rand Index (ARI) Scores:")
print(ari_table)

for dim in dbscan_dims:
    # Perform PCA
    pca = PCA(n_components=dim)
    test_PCA = pca.fit_transform(X_)

    # Run DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_neighbors)
    labels = dbscan.fit_predict(test_PCA)

    # Compute ARI
    ari = adjusted_rand_score(y_, labels)
    ari_scores.append(ari)

    # Create a DataFrame for visualization
    testDF = pd.DataFrame(data=test_PCA[:, :2], columns=["x-axis", "y-axis"])
    testDF["Label"] = labels

    # Plot the results
    sns.lmplot(x="x-axis", y="y-axis", hue="Label", data=testDF, fit_reg=False, height=8)
    plt.title(f"DBSCAN Clustering Result (Dimension = {dim})")
    plt.grid()
    plt.show()

    # Perform t-SNE on the reduced dimensions
    tSNE = TSNE(n_components=2, random_state=42)
    X_tSNE = tSNE.fit_transform(test_PCA)

    # Randomly choose 100 samples for visualization
    random_indices = np.random.choice(X_tSNE.shape[0], size=100, replace=False)
    X_tSNE_sample = X_tSNE[random_indices]
    y_sample = y_[random_indices]

    # Create a DataFrame for visualization
    tSNE_df = pd.DataFrame(data=X_tSNE_sample, columns=["x-axis", "y-axis"])
    tSNE_df["Label"] = y_sample

    # Plot the t-SNE visualization
    sns.lmplot(x="x-axis", y="y-axis", hue="Label", data=tSNE_df, fit_reg=False, height=8)
    plt.title(f"t-SNE Visualization (Dimension = {dim})")
    plt.grid()
    plt.show()

# Create a table for ARI scores
ari_table = pd.DataFrame(data={"Dimension": dbscan_dims, "ARI Score": ari_scores})

print("Adjusted Rand Index (ARI) Scores:")
print(ari_table)
