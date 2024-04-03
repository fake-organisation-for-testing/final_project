from sklearn.cluster import KMeans

def train_model(data):
    model = KMeans(n_clusters=3, n_init=15)
    model.fit(data)
    return model