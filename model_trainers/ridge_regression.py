from sklearn.linear_model import Ridge

def train_model(X_train, y_train):
    model = Ridge(alpha = 0.5)
    model.fit(X_train, y_train)
    return model