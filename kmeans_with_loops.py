import numpy as np
import matplotlib.pyplot as plt

def get_centroid(points, cluster_assignments,k):
        return np.array([np.mean(points[cluster_assignments == i], axis=0) for i in range(k)])
              
def assign_clusters(cluster_assignments, centroids, points):

    for i, p in enumerate(points):
        distances = []
        for centroid in centroids:
            distance = get_eu_distance(p, centroid)
            distances.append(distance)
        cluster_assignments[i] = np.argmin(distances)

    return cluster_assignments

def get_eu_distance(a, b):
        return np.sqrt(sum((a-b)**2))

                          
if __name__ == "__main__":
    n_samples = 1000
    n_features = 5
    k = 5
    max_iters = 100

    points = np.random.randn(n_samples, n_features)
    cluster_assignments = np.zeros(n_samples, dtype=int)
    init_centroid_indices = np.random.choice(n_samples, size=k, replace=False)
    centroids = points[init_centroid_indices]

    max_dist = 0
    dist_from_centroid = [get_eu_distance(centroids[c],p) for c,p in zip(cluster_assignments, points)]
    distances = []
    for t in range(max_iters):
        
        cluster_assignments = assign_clusters(cluster_assignments, centroids, points)
        new_centroids = get_centroid(points, cluster_assignments,k)
        
        dist_from_centroid = [get_eu_distance(new_centroids[c],p) for c,p in zip(cluster_assignments, points)]
        print(f"At t={t}, dist_from_centroid", sum(dist_from_centroid))  
        distances.append(sum(dist_from_centroid))
        if np.all(new_centroids == centroids):
             print("CONVERGENCE")
             break
        centroids = new_centroids
    test_samples = 5
    new_points = np.random.randn(test_samples, n_features)
    inference_cluster_assignments = np.zeros(test_samples)
    inference_cluster_assignments = assign_clusters(inference_cluster_assignments, centroids, new_points)
    print("inference_cluster_assignments", inference_cluster_assignments)

    
    plt.plot(distances)
    plt.xlabel("Iteration")
    plt.title("K-Means Convergence")
    plt.ylabel("SSE")
    plt.show()




