import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import leaders

# Example data
data = {
    'Asset1': [0.04, 0.03, 0.02, 0.03, 0.05],
    'Asset2': [0.03, 0.02, 0.01, 0.02, 0.04],
    'Asset3': [0.06, 0.05, 0.04, 0.05, 0.07],
    'Asset4': [0.02, 0.01, 0.02, 0.01, 0.03]
}

returns_df = pd.DataFrame(data)

# Calculate covariance matrix using Ledoit-Wolf shrinkage
cov_matrix = LedoitWolf().fit(returns_df).covariance_

# Calculate pairwise distances for hierarchical clustering
distance_matrix = pdist(returns_df, metric='euclidean')

# Perform hierarchical clustering
linkage_matrix = linkage(distance_matrix, method='single')

# Determine the number of clusters
num_clusters = 2

# Assign assets to clusters
asset_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# Calculate the inverse volatility of each cluster
clustered_cov_matrix = np.zeros_like(cov_matrix)
for cluster in range(1, num_clusters + 1):
    assets_in_cluster = returns_df.columns[asset_labels == cluster]
    sub_cov_matrix = cov_matrix.loc[assets_in_cluster, assets_in_cluster].values
    total_volatility = np.sqrt(np.diag(sub_cov_matrix).sum())
    clustered_cov_matrix[assets_in_cluster, assets_in_cluster] = 1 / total_volatility

# Calculate portfolio weights using the HRP approach
portfolio_weights = clustered_cov_matrix / clustered_cov_matrix.sum()

print("Portfolio Weights:")
print(portfolio_weights)
