import numpy as np
import matplotlib.pyplot as plt
from ml.linear_regression import LinearRegression

# Generazione dati sintetici
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(100)

# Preprocessing
X = X.reshape(-1, 1)

# Allenamento modello
model = LinearRegression()
model.fit(X, y)

# Predizione
y_pred = model.predict(X)

# Visualizzazione
plt.scatter(X, y, color='blue', label='Dati reali')
plt.plot(X, y_pred, color='red', label='Predizione')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Regressione Lineare")
plt.show()

# Output dei parametri
print("Peso (coefficiente angolare):", model.weights)
print("Bias (intercetta):", model.bias)
print("Loss (MSE):", model.loss(X, y))
