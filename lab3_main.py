import tensorflow_datasets as tfds

# Завантажуємо дані
(train_ds, val_ds), ds_info = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:]'], # Розділяємо на 80% навчання та 20% перевірки
    as_supervised=True,
    with_info=True
)

import tensorflow as tf

def format_image(image, label):
    image = tf.image.resize(image, (180, 180)) / 255.0
    return image, label

train_batches = train_ds.shuffle(1000).map(format_image).batch(32)
val_batches = val_ds.map(format_image).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(5, activation='softmax') # 5 класів квітів
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_batches, validation_data=val_batches, epochs=20)

import matplotlib.pyplot as plt
import numpy as np

# Отримуємо назви класів (маргаритка, троянда тощо)
class_names = ds_info.features['label'].names

# Беремо один батч (32 фото) з валідаційних даних
image_batch, label_batch = next(iter(val_batches))
predictions = model.predict(image_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i])

    predicted_label = np.argmax(predictions[i])
    actual_label = label_batch[i]

    color = 'green' if predicted_label == actual_label else 'red'
    plt.title(f"Передбачено: {class_names[predicted_label]}\nНасправді: {class_names[actual_label]}", color=color)
    plt.axis("off")

plt.show()