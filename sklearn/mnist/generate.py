from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from joblib import dump

data = fetch_openml("mnist_784", parser="auto")
X = data.data
y = data.target
model = MLPClassifier(random_state=42)
model.fit(X, y)
dump(model, "model.joblib")
