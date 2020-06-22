import numpy as np
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
from utils import fill_data, normalize_data

# https://github.com/yiuhyuk/knn/blob/master/k-means.ipynb


data = pd.read_csv('data/acath.csv', sep=';')
data = fill_data(data)
data = data[['sex', 'age', 'cad.dur', 'choleste', 'sigdz']]
print(data)
#Sorting ensures that I will pick k observations that are all very similar to each other
# as my initial centroids. This way, the starting centroids will be suboptimal and we can
# more clearly see how the algorithm is able to converge to much better centroids (and clusters)
# data.sort_values(by=['age', 'cad.dur', 'choleste'], inplace=True)
cluster_array = np.array(data)

# Calculate Euclidean distance between two observations
def calc_distance(X1, X2):
    return (sum((X1 - X2)**2))**0.5

# Assign cluster clusters based on closest centroid
def assign_clusters(centroids, cluster_array):
    clusters = []
    for i in range(cluster_array.shape[0]):
        distances = []
        for centroid in centroids:
            distances.append(calc_distance(centroid,
                                           cluster_array[i]))
        cluster = [z for z, val in enumerate(distances) if val==min(distances)]
        #tells to which cluster it should be asssigned
        clusters.append(cluster[0])
    return clusters

# Calculate new centroids based on each cluster's mean
def calc_centroids(clusters, cluster_array):
    new_centroids = []
    cluster_df = pd.concat([pd.DataFrame(cluster_array),
                            pd.DataFrame(clusters,
                                         columns=['cluster'])],
                           axis=1)
    for c in set(cluster_df['cluster']):
        current_cluster = cluster_df[cluster_df['cluster']\
                                     ==c][cluster_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids

# Calculate variance within each cluster
def calc_centroid_variance(clusters, cluster_array):
    sum_squares = []
    cluster_df = pd.concat([pd.DataFrame(cluster_array),
                            pd.DataFrame(clusters,
                                         columns=['cluster'])],
                           axis=1)
    for c in set(cluster_df['cluster']):
        current_cluster = cluster_df[cluster_df['cluster']\
                                     ==c][cluster_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        mean_repmat = np.matlib.repmat(cluster_mean,
                                       current_cluster.shape[0],1)
        sum_squares.append(np.sum(np.sum((current_cluster - mean_repmat)**2)))
    return sum_squares

k = 3
cluster_vars = []
#cluster_array tiene la data, aca vamos a generar los centroids de forma aleatoria tomando
centroids = [cluster_array[i+2] for i in range(k)]
clusters = assign_clusters(centroids, cluster_array)
initial_clusters = clusters
print(0, round(np.mean(calc_centroid_variance(clusters, cluster_array))))
for i in range(50):
    centroids = calc_centroids(clusters, cluster_array)
    clusters = assign_clusters(centroids, cluster_array)
    print(clusters)
    cluster_var = np.mean(calc_centroid_variance(clusters,
                                                 cluster_array))
    cluster_vars.append(cluster_var)
    print(i+1, round(cluster_var))

plt.subplots(figsize=(9,6))
plt.plot(cluster_vars)
plt.xlabel('Iterations')
plt.ylabel('Mean Sum of Squared Deviations');
plt.savefig('mean_ssd', bpi=150)

print(cluster_array[1, :])

plt.subplots(figsize=(9,6))
plt.scatter(x=cluster_array[:,4], y=cluster_array[:,1],
            c=initial_clusters, cmap=plt.cm.Spectral);
plt.xlabel('Colesterol')
plt.ylabel('enfermo');
plt.savefig('initial_clusters', bpi=150)

plt.subplots(figsize=(9,6))
plt.scatter(x=cluster_array[:,4], y=cluster_array[:,1],
            c=clusters, cmap=plt.cm.Spectral);
plt.xlabel('Colesterol')
plt.ylabel('enfermo');
plt.savefig('final_clusters', bpi=150)