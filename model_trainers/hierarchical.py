import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

def train_model(data):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
    now = cluster.fit_predict(reduced_data)
    return now