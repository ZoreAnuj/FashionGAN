# FashionGAN Tutorial

This repository contains a step-by-step tutorial on building and training a Fashion Generative Adversarial Network (FashionGAN) using TensorFlow. FashionGAN is a powerful AI model that generates synthetic fashion images resembling real clothing, shoes, and accessories.

## Results: [Link](https://drive.google.com/drive/folders/1-4fpju13NJEO-9oqz_4P1IWAk1Liw69i?usp=sharing)

## Table of Content

1. Introduction
2. Installation
3. Import Dependencies and Data
4. Visualize Data and Build Dataset
5. Build Neural Network
6. Construct Training Loop
7. Review Performance
8. Test Out the Generator
9. Save the Model

### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

To run the code in this tutorial, you need to have TensorFlow and TensorFlow Datasets installed. You can install them using pip:

```bash
pip install tensorflow tensorflow-datasets

```
### Import Dependencies and Data
```bash
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
```
### Visualize Data and Build Dataset

In this section, we load the Fashion MNIST dataset using the TensorFlow Datasets API and visualize a few sample images. We then preprocess the data, scale the images, and create a TensorFlow dataset for training.

```bash
# Load the Fashion MNIST dataset
ds = tfds.load('fashion_mnist', split='train')

# Create an iterator to access the data
dataiterator = ds.as_numpy_iterator()

# Visualize a few sample images
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4):
    sample = dataiterator.next()
    ax[idx].imshow(sample['image'])
    ax[idx].title.set_text(sample['label'])

# Preprocess the data and create a TensorFlow dataset
def scale_images(data):
    image = data['image']
    return image / 255

ds = tfds.load('fashion_mnist', split='train')
ds = ds.map(scale_images)
ds = ds.cache()
ds = ds.shuffle(60000)
ds = ds.batch(128)
ds = ds.prefetch(64)

```

### Build Neural Network

Now, let's build the neural network architecture for our FashionGAN. The model consists of a generator and a discriminator. The generator takes random noise as input and generates fake fashion images, while the discriminator acts as a binary classifier to distinguish between real and fake images.

```bash
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D

# Build the generator
def build_generator():
    model = Sequential()
    model.add(Dense(7*7*128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))
    # Add more layers...

    return model

# Build the discriminator
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(32, 5, input_shape = (28,28,1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    # Add more layers...

    return model

# Create instances of the generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

```

### Construct Training Loop
In this section, we define the training loop for our FashionGAN. The generator and discriminator play a game of cat and mouse, constantly trying to outwit each other. We use binary cross-entropy loss and Adam optimizer for both the generator and discriminator.

```bash
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Define loss and optimizer
g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

# Create a subclassed model for FashionGAN
class FashionGAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(FashionGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss, d_loss):
        super(FashionGAN, self).compile()
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)
        # Training steps...

```

### Review Performance

After the training loop, we evaluate the performance of our FashionGAN by plotting the loss values of the generator and discriminator. Lower generator loss indicates better image generation.

```bash
# Training the FashionGAN
fashgan = FashionGAN(generator, discriminator)
fashgan.compile(g_opt, d_opt, g_loss, d_loss)
hist = fashgan.fit(ds, epochs=2000)

# Review performance
plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='d_loss')
plt.plot(hist.history['g_loss'], label='g_loss')
plt.legend()
plt.show()

```

### Test Out the Generator

Finally, we can test the trained generator by generating new fashion images from random noise.

```bash
# Generating new fashion images
imgs = generator.predict(tf.random.normal((16, 128, 1)))

# Display the generated images
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(10,10))
for r in range(4):
    for c in range(4):
        ax[r][c].imshow(imgs[(r+1)*(c+1)-1]

```
### Save the Model

To keep the creative magic alive, we save the trained generator and discriminator models for future use.

```bash
# Save the models
generator.save('generator.h5')
discriminator.save('discriminator.h5')
```
Congratulations! You have now mastered the art of Fashion

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
