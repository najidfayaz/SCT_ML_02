# SCT_ML_02
Customer Segmentation using K-Means Clustering
This project applies K-Means Clustering to segment customers of a retail store based on their Annual Income and Spending Score.
The goal is to group similar customers together for better marketing strategies.

Dataset
The dataset used is Mall_Customers.csv containing the following features:

CustomerID: Unique ID for each customer
Gender: Male/Female
Age: Age of the customer
Annual Income (k$): Income in thousand dollars
Spending Score (1-100): Score assigned by the mall based on customer behavior
Project Workflow
Import Libraries
Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
Load Dataset
Load CSV file either from local system or Google Colab
Data Exploration
Check for null values, dataset shape, and basic statistics
Feature Selection
Select Annual Income (k$) and Spending Score (1-100) for clustering
Data Preprocessing
Standardize features using StandardScaler
Elbow Method
Determine the optimal number of clusters using WCSS
K-Means Clustering
Train model and assign cluster labels to each customer
Visualization
Scatter plot of clusters
Centroid visualization
Box plots showing feature distributions per cluster
Visualizations
Elbow Method Plot: Helps choose optimal k for K-Means
Cluster Plot: Scatter plot with different colors for each cluster
Centroid Labels: Red X markers with cluster names
Box Plots: Show distribution of Annual Income and Spending Score per cluster
Technologies Used
Python 
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
