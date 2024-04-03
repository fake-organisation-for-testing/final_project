from sklearn.metrics import precision_score

def calculate_metric(y_test, y_pred):
    metric = precision_score(y_test, y_pred)
    return metric