from sklearn.neighbors import KNeighborsClassifier

def train_model(X_train, y_train):
    print("training KNeighborsClassifier model")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    return model