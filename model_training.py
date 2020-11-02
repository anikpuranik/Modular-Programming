def train_model(ModelClass, X_train, X_test, y_train, y_test, **kwargs):
    model = ModelClass(**kwargs)
    model.fit(X_train, y_train)
    accuracy_score = round(model.score(X_test, y_test) * 100, 2)    
    return model, accuracy_score