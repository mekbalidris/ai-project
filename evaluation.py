import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def evaluate_clustering(model, x_train, k):
    print(f"evaluating for k={k}...")
    labels = model.labels_
    
    inertia = model.inertia_
    sil = silhouette_score(x_train, labels)
    db = davies_bouldin_score(x_train, labels)
    
    print(f"inertia: {inertia:.2f}")
    print(f"silhouette score: {sil:.4f}")
    print(f"davies bouldin index: {db:.4f}")

def compare_k_values(x_train):
    print("comparing k values...")
    print("k   inertia   silhouette   db_index")
    
    # comparing k from 2 to 5
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(x_train)
        labels = km.labels_
        sil = silhouette_score(x_train, labels)
        db = davies_bouldin_score(x_train, labels)
        print(f"{k}   {km.inertia_:.2f}   {sil:.3f}   {db:.3f}")

def visualize_results(model, x_train):
    print("visualizing clusters...")
    
    # using pca to reduce to 2 dimensions for plotting
    pca = PCA(n_components=2)
    
    # FIX: use .values to convert to simple array so we don't get the warning
    x_pca = pca.fit_transform(x_train.values)
    centers_pca = pca.transform(model.cluster_centers_)
    
    plt.figure(figsize=(10, 6))
    
    # plot the patients
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=model.labels_, cmap='viridis', alpha=0.6)
    
    # plot the centers
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, marker='X', label='centroids')
    
    plt.title("cluster visualization (pca)")
    plt.legend()
    plt.savefig('5_cluster_pca.png')
    plt.close()
    
    print("saved cluster plot.")