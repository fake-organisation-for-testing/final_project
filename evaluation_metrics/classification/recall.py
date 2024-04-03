from sklearn.metrics import recall_score

def calculate_metric(y_test, y_pred):
    metric = recall_score(y_test, y_pred)
    return metric