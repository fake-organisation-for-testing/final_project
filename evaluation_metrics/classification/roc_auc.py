from sklearn.metrics import roc_auc_score

def calculate_metric(y_test, y_pred):
    metric = roc_auc_score(y_test, y_pred)
    return metric