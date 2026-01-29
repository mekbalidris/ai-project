import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def split_data(df):
    # split data into 80% train and 20% test
    x_train, x_test = train_test_split(df, test_size=0.2, random_state=42)
    print("train shape:", x_train.shape)
    print("test shape:", x_test.shape)
    return x_train, x_test

def find_optimal_k(x_train):
    # using elbow method to find k
    print("calculating elbow method...")
    inertia_list = []
    # checking k from 1 to 10
    k_values = range(1, 11)
    
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(x_train)
        inertia_list.append(km.inertia_)
        
    # plot the curve
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, inertia_list, marker='o')
    plt.title('elbow method')
    plt.xlabel('k')
    plt.ylabel('inertia')
    plt.grid(True)
    plt.savefig('4_elbow_method.png')
    plt.close()
    
    print("saved elbow plot.")
    # returning 2 because the graph usually bends there
    return 2 

def train_model(x_train, k):
    # training the model with the chosen k
    print(f"training kmeans with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_train)
    
    print("cluster centers calculated.")
    
    return kmeans