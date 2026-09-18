import numpy as np
from collections import Counter

# Training data: 9 points across 3 classes, and query point X in the center
X_train = np.array([
    [4.0, 5.0], [3.0, 6.0], [2.0, 5.0],  # blue (points 1, 2, 3)
    [6.0, 6.0], [5.5, 6.2], [7.0, 7.0],  # green (points 4, 5, 6)
    [5.0, 3.2], [4.2, 3.5], [5.0, 2.0]   # red (points 7, 8, 9)
])

y_train = np.array([
    "blue", "blue", "blue",
    "green", "green", "green",
    "red", "red", "red"
])

X = np.array([5.0, 5.0])

# One-liner KNN prediction function (Vectorized Euclidean Distance + Majority Vote)
def predict_knn(X_train, y_train, x, k):
    return Counter(y_train[np.argsort(np.linalg.norm(X_train - x, axis=1))[:k]]).most_common(1)[0][0]

if __name__ == "__main__":
    print(f"Query point X: {tuple(X.tolist())}")
    try:
        k = int(input(f"Enter K value (1-{len(X_train)}): "))
        if not (1 <= k <= len(X_train)):
            print(f"Error: K must be between 1 and {len(X_train)}")
            exit(1)

        winner = predict_knn(X_train, y_train, X, k)
        print(f">> Prediction for K = {k}: '{winner}'")
    except ValueError:
        print("Error: Please enter a valid integer.")
