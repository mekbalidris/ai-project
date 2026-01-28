import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def evaluate_clustering(model, x_train, k):
    # calculate metrics
    print(f"evaluating for k={k}...")
    labels = model.labels_
    
    inertia = model.inertia_
    sil = silhouette_score(x_train, labels)
    db = davies_bouldin_score(x_train, labels)
    
    print(f"inertia: {inertia}")
    print(f"silhouette score: {sil}")
    print(f"davies bouldin index: {db}")

def compare_k_values(x_train):
    # compare different k values
    print("comparing k values...")
    print("k   inertia   silhouette   db_index")
    
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42).fit(x_train)
        labels = km.labels_
        sil = silhouette_score(x_train, labels)
        db = davies_bouldin_score(x_train, labels)
        # simple print format
        print(f"{k}   {km.inertia_:.2f}   {sil:.3f}   {db:.3f}")

def visualize_results(model, x_train):
    # use pca to make it 2d for plotting
    print("visualizing clusters...")
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_train)
    centers_pca = pca.transform(model.cluster_centers_)
    
    plt.figure(figsize=(10, 6))
    # plot the points
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=model.labels_, cmap='viridis', alpha=0.6)
    # plot the centers
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, marker='X')
    plt.title("cluster visualization (pca)")
    plt.savefig('5_cluster_pca.png')
    plt.close()
    print("saved cluster plot.")