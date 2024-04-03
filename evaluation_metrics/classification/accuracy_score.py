from sklearn.metrics import accuracy_score

def calculate_metric(y_test, y_pred):
    metric = accuracy_score(y_test, y_pred)
    return metric