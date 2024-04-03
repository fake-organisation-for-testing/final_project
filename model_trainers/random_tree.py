from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_features=0.5,
        max_depth=3,
        min_samples_split=2,
        random_state=0
    )

    model.fit(X_train, y_train)
    return model