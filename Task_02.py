import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 1. Load the dataset
try:
    df = pd.read_csv('MallData.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("ERROR: 'MallData.csv' not found. Please check the file name and location.")
    sys.exit()

# 2. Select Features
# use annual income (k$) and spending score (1-100) for clustering
X = df.iloc[:, [3, 4]].values

# 3. apply K-Means Clustering
# use 5 clusters 
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 4. Visualization
plt.figure(figsize=(10, 6))

# Plotting the 5 Clusters
# C1
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
# C2
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
# C3
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
# C4
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
# C5
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')

# Plotting the Centroids of each group
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()