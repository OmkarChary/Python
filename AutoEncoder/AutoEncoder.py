# Autoencoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], 784)) / 255.0
test_images = test_images.reshape((test_images.shape[0], 784)) / 255.0

class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(784, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate and compile autoencoder model
autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train autoencoder
history = autoencoder.fit(train_images, train_images, epochs=10, batch_size=256, validation_data=(test_images, test_images))

# Test autoencoder
random_test_index = np.random.randint(test_images.shape[0])
random_test_image = test_images[random_test_index]
reconstructed_image = autoencoder.predict(random_test_image.reshape(1, 784))

print('Original Image:')
plt.imshow(random_test_image.reshape(28, 28), cmap='gray')
plt.show()

print('Reconstructed Image:')
plt.imshow(reconstructed_image.reshape(28, 28), cmap='gray')
plt.show()

# Evaluate autoencoder
loss = autoencoder.evaluate(test_images, test_images, verbose=0)
print('Autoencoder Loss:', loss)

# Lower the losses
print('Loss History:', history.history['loss'])

# Improve on the losses
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
history = autoencoder.fit(train_images, train_images, epochs=10, batch_size=256, validation_data=(test_images, test_images))

print('Improved Loss History:', history.history['loss'])

# Check the output for the improved losses
reconstructed_image = autoencoder.predict(random_test_image.reshape(1, 784))

print('Improved Reconstructed Image:')
plt.imshow(reconstructed_image.reshape(28, 28), cmap='gray')
plt.show()








