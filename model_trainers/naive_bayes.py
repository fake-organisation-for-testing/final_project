from sklearn.naive_bayes import GaussianNB

def train_model(X_train, y_train):
    print("training naive_bayes model")
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model