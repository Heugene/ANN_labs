from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
# Завантаження набору даних Iris
data = load_iris()
X = data.data  # Вхідні дані (ознаки квітки)
y = data.target  # Мітки класів (види квіток)
# Розділення даних на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Створення моделі
model = Sequential()
# Додавання прихованого шару з 10 нейронами та активацією ReLU
model.add(Dense(10, input_shape=(4,), activation='relu'))
# Додавання вихідного шару для класифікації на 3 класи
model.add(Dense(3, activation='softmax'))
# Компіляція моделі
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Навчання моделі
model.fit(X_train, y_train, epochs=50, batch_size=5, validation_split=0.1)
# Оцінка точності моделі на тестових даних
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Точність на тестових даних: {test_acc * 100:.2f}%")