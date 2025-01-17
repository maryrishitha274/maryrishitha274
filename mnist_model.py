import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  
    keras.layers.Dense(128, activation='relu'),  
    keras.layers.Dropout(0.2),                   
    keras.layers.Dense(10)                       
])

model.compile(
    optimizer='adam', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy']
)


model.fit(train_images, train_labels, epochs=5)


loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test Accuracy: {accuracy:.2f}')

predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)


plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i], cmap='gray')
    plt.title(f'True: {test_labels[i]}\nPred: {predicted_classes[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
