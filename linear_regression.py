class LinearRegression:
    def __init__(self, learning_rate = 0.01, n_iters = 10):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n, f = X.shape
        self.weights = np.random.randn(X.shape[1])
        self.bias = np.random.randn(1)

        for _ in range(self.n_iters):
            # compute y = mx + c
            y_pred = np.dot(X, self.weights) + self.bias

            # compute grads
            error = y_pred - y
            dw = (2/n) * (np.dot(X.T, error))
            db = (2/n) * (np.sum(error))

            # update params
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    

if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    X = np.random.rand(50, 1)
    y = np.random.uniform(0, 1, size=50)
    iterations = 1000
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
    print(" X_train, X_test, y_train, y_test",  X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    model = LinearRegression(learning_rate=0.01, n_iters=iterations)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("MSE", mse)

    y_pred_line = model.predict(X)
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color="blue", label="training")
    m2 = plt.scatter(X_test, y_test, color="red", label = "testing")
    plt.plot(X, y_pred_line, color="black", label="Prediction")
    plt.title(f"Linear Regression: Iters = {iterations}, MSE: {round(mse, 4)}")
    plt.show()
    plt.close()
    exit(0)



