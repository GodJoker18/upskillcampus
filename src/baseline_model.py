import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_and_evaluate(X, y):
    # Flatten sequences
    X = X.reshape(X.shape[0], -1)

    # Proper random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=50,      # reduced for speed
        max_depth=20,         # controls overfitting
        random_state=42,
        n_jobs=-1             # use all CPU cores
    )

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse
