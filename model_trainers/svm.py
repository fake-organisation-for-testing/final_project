from sklearn.svm import SVC

def train_model(X_train, y_train):
    print("training svm model")
    model = SVC(C = 5)
    model.fit(X_train, y_train)

    return model