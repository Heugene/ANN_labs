import numpy as np
import matplotlib.pyplot as plt

# Клас Perceptron залишається таким самим, як у попередньому прикладі
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation >= 0 else 0

    def train(self, training_inputs, labels, epochs=100):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

# 1. Готуємо дані для XOR
raw_inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
labels = np.array([0, 1, 1, 0])

# 2. Додаємо нову ознаку: x1 * x2
# Це перетворює [x1, x2] на [x1, x2, x1*x2]
extra_feature = (raw_inputs[:, 0] * raw_inputs[:, 1]).reshape(-1, 1)
xor_inputs = np.hstack((raw_inputs, extra_feature))

# 3. Навчаємо на 3-х входах
perceptron = Perceptron(input_size=3, learning_rate=0.2)
perceptron.train(xor_inputs, labels, epochs=100)

# 4. Перевірка
print("Результати для XOR (з дод. ознакою):")
for i, inputs in enumerate(xor_inputs):
    res = perceptron.predict(inputs)
    print(f"Вхід: {raw_inputs[i]} (ознака {inputs[2]}) -> Прогноз: {res}")

# 5. Візуалізація (3D)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Малюємо точки
ax.scatter(xor_inputs[:, 0], xor_inputs[:, 1], xor_inputs[:, 2],
           c=labels, cmap='bwr', s=100, edgecolors='k')

# Малюємо площину рішення
xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 10), np.linspace(-0.2, 1.2, 10))
# Рівняння: w1*x + w2*y + w3*z + w0 = 0  => z = -(w1*x + w2*y + w0) / w3
if perceptron.weights[3] != 0:
    zz = -(perceptron.weights[1] * xx + perceptron.weights[2] * yy + perceptron.weights[0]) / perceptron.weights[3]
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='green')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X1 * X2')
plt.title("XOR у 3D просторі ознак")
plt.show()