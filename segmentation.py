import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("customers.csv")

# Select important features
X = df[["AnnualIncome", "SpendingScore"]]

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=0)
df["Cluster"] = kmeans.fit_predict(X)

# Print data with clusters
print(df)

# Plot clusters
plt.scatter(df["AnnualIncome"], df["SpendingScore"], c=df["Cluster"])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()