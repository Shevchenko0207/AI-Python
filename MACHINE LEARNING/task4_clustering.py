import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_dataset(file_path):
    """
    Load a dataset from a CSV file.
    """
    try:
        dataset = pd.read_csv(file_path)
    except FileNotFoundError:
        # Якщо файлу немає, створюємо синтетичні дані з 3 вираженими кластерами
        print(f"[Warning] {file_path} not found. Generating sample data...")
        np.random.seed(42)

        # Генеруємо 3 різні хмари точок
        cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(100, 2))
        cluster2 = np.random.normal(loc=[8, 3], scale=0.6, size=(100, 2))
        cluster3 = np.random.normal(loc=[5, 7], scale=0.5, size=(100, 2))

        all_data = np.vstack([cluster1, cluster2, cluster3])

        dataset = pd.DataFrame(all_data, columns=["feature1", "feature2"])
        dataset.to_csv(file_path, index=False)

    return dataset


def preprocess_data(dataset, selected_features):
    """
    Scales selected features from the given dataset.
    """
    # Виділяємо потрібні ознаки
    X = dataset[selected_features]

    # Застосовуємо StandardScaler для нормалізації
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def apply_kmeans_clustering(X_scaled, num_clusters):
    """
    Apply K-means clustering to scaled data.
    """
    # Ініціалізуємо та тренуємо модель K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    return clusters, kmeans.cluster_centers_


def visualize_clusters(X_scaled, clusters, cluster_centers):
    """
    Visualize the resulting clusters and save the image.
    """
    plt.figure(figsize=(8, 6))

    # Малюємо точки, розфарбовуючи їх відповідно до визначеного кластера
    plt.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=clusters,
        cmap="viridis",
        alpha=0.6,
        edgecolors="k",
        label="Data Points",
    )

    # Виділяємо центри кластерів (центроїди) великими червоними хрестиками
    plt.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        c="red",
        marker="X",
        s=200,
        label="Centroids",
    )

    plt.title("K-Means Clustering Results")
    plt.xlabel("Scaled Feature 1")
    plt.ylabel("Scaled Feature 2")
    plt.legend()
    plt.tight_layout()

    # Замість plt.show() автоматично зберігаємо результат локально
    plt.savefig("kmeans_clusters.png", dpi=300)
    plt.close()
    print("[Success] Cluster visualization saved as 'kmeans_clusters.png'.")

    return None


# ==========================================
# EXECUTION FLOW
# ==========================================

# Load the dataset
file_path = "features.csv"
data = load_dataset(file_path)

# Selected features for clustering
selected_features = ["feature1", "feature2"]

# Preprocess the data
X_scaled = preprocess_data(data, selected_features)

# Choose the number of clusters (K)
num_clusters = 3

# Apply K-Means clustering
clusters, cluster_centers = apply_kmeans_clustering(X_scaled, num_clusters)

# Visualize the clusters
visualize_clusters(X_scaled, clusters, cluster_centers)
