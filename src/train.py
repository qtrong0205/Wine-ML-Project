def train(model, X_train, y_train, **fit_params):
    model.fit(X_train, y_train, **fit_params)
    return model
