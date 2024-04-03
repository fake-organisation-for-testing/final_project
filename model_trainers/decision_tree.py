from sklearn.tree import DecisionTreeClassifier

def train_model(X_train, y_train):
    ## gini
    decision_gini = DecisionTreeClassifier()
    #cut the tree based on how tall the tree is
    decision_depth = DecisionTreeClassifier(max_depth=5)
    #check for uncertainty
    decision_entropy = DecisionTreeClassifier(criterion='entropy')

    decision_gini.fit(X_train, y_train)
    decision_depth.fit(X_train, y_train)
    decision_entropy.fit(X_train, y_train)
    return {"decision_gini": decision_gini, "decision_depth": decision_depth, "decision_entropy": decision_entropy}