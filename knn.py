import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]

def knn_predict_batch(training_data,training_labels,test_data,k):

    predictions = []
    for point in test_data:
        prediction = knn_predict(training_data,training_labels,point,k)
        predictions.append(prediction)

    return np.array(predictions)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

if __name__ == "__main__":
    n_samples = 500
    # 3 classes with some overlap
    # This makes K actually matter.
    class_A = np.random.randn(n_samples // 3, 2) + np.array([-2, -2])
    class_B = np.random.randn(n_samples // 3, 2) + np.array([2, 2])
    class_C = np.random.randn(n_samples - 2 * (n_samples // 3), 2) + np.array([2, -2])

    X = np.vstack([
        class_A,
        class_B,
        class_C
    ])

    y = np.array(
        ['A'] * len(class_A) +
        ['B'] * len(class_B) +
        ['C'] * len(class_C)
    )
    n = len(X)

    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    print("Training samples   :", len(X_train))
    print("Validation samples :", len(X_val))
    print("Test samples       :", len(X_test))
    k_values = range(3, 25)

    train_accuracies = []
    val_accuracies = []

    for k in k_values:
        print("K:",k)
        # Training accuracy
        train_predictions = knn_predict_batch(X_train,y_train,X_train,k)
        train_acc = accuracy(y_train,train_predictions)

        # Validation accuracy
        val_predictions = knn_predict_batch(X_train,y_train,X_val,k)
        val_acc = accuracy(y_val,val_predictions)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    print(
        f"K = {k}"
        f"Train Accuracy = {train_acc} "
        f"Validation Accuracy = {val_acc}"
    )

    best_index = np.argmax(val_accuracies)
    best_k = list(k_values)[best_index]
    best_val_accuracy = val_accuracies[best_index]

    print("Best K:", best_k)
    print("Best Validation Accuracy:", best_val_accuracy)

    plt.figure(figsize=(9, 5))

    plt.plot(
        k_values,
        train_accuracies,
        marker="o",
        label="Train Accuracy"
    )

    plt.plot(
        k_values,
        val_accuracies,
        marker="o",
        label="Validation Accuracy"
    )

    plt.axvline(
        best_k,
        linestyle="--",
        label=f"Best K = {best_k}"
    )

    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("KNN: Accuracy vs K")
    plt.xticks(list(k_values))
    plt.show()



