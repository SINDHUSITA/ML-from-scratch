class LogisticRegression:
    def __init__(self, n_iters=10, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
    
    def fit(self, X, y):
        n, c = X.shape
        
        self.W = np.random.randn(c)
        self.b = np.random.rand(1)

        for i in range(self.n_iters):
            # print("No. of iters: ", i)
            # predictions
            regression_vals = np.dot(X, self.W) + self.b
            y_pred = self.sigmoid(regression_vals)
            
            # gradients
            error = y_pred - y
            dw = (2/n) * (np.dot(X.T, error))
            db = (2/n) * (np.sum(error))

            # update params
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db
    
    def predict(self, X):
        regression_vals = np.dot(X, self.W) + self.b
        sigmoid_vals = self.sigmoid(regression_vals)

        y_pred_class = [1 if i > 0.5 else 0 for i in sigmoid_vals]
        return np.array(y_pred_class)

    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))

if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    def calc_accuracy(y_pred, y_true):
        return (np.sum(y_pred == y_true)) / len(y_true)
    
    iters = [10, 20, 50, 80, 100, 125, 150, 175, 200, 300, 350, 500]
    accuracies = []
    X = np.random.randn(50, 2)
    y = np.random.randint(0,1,size=(50))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

    for iter in iters:
        model = LogisticRegression(n_iters=iter, learning_rate=0.01)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = calc_accuracy(y_pred, y_test)
        accuracies.append(accuracy)
        print("Accuracy:iter: ", accuracy, iter)
    
    fig = plt.plot(iters, accuracies)
    plt.title("No. of iterations vs Accuracy:")
    plt.show()


    


