from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

import matplotlib as mpl

from IPython.display import clear_output
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image


class GanDeepdreamService:
    def __init__(self):
        self.url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
        self.original_img = ''
        pass

    def run_deepdream(self):
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        # Maximize the activations of these layers
        names = ['mixed3', 'mixed5']
        layers = [base_model.get_layer(name).output for name in names]

        # Create the feature extraction model
        dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
        original_img = self.download(self.url, target_size=[225, 375])
        original_img = np.array(original_img)
        dream_img = self.run_deep_dream_simple(model=dream_model, img=original_img,
                                          steps=800, step_size=0.001)


    @tf.function
    def deepdream(self, model, img, step_size):
        with tf.GradientTape() as tape:
            # This needs gradients relative to `img`
            # `GradientTape` only watches `tf.Variable`s by default
            tape.watch(img)
            loss = self.calc_loss(img, model)

        # Calculate the gradient of the loss with respect to the pixels of the input image.
        gradients = tape.gradient(loss, img)

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
        # You can update the image by directly adding the gradients (because they're the same shape!)
        img = img + gradients*step_size
        img = tf.clip_by_value(img, -1, 1)

        return loss, img

    def run_deep_dream_simple(self, model, img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
        img = tf.keras.applications.inception_v3.preprocess_input(img)

        for step in range(steps):
            loss, img = self.deepdream(model, img, step_size)
            if step % 100 == 0:
                clear_output(wait=True)
                self.show(self.deprocess(img))
                print ("Step {}, loss {}".format(step, loss))

        result = self.deprocess(img)
        clear_output(wait=True)
        self.show(result)
        return result

    def calc_loss(img, model):
        # Pass forward the image through the model to retrieve the activations.
        # Converts the image into a batch of size 1.
        img_batch = tf.expand_dims(img, axis=0)
        layer_activations = model(img_batch)

        losses = []
        for act in layer_activations:
            loss = tf.math.reduce_mean(act)
            losses.append(loss)

        return  tf.reduce_sum(losses)

    # Download an image and read it into a NumPy array.
    def download(self, url, target_size=None):
        name = url.split('/')[-1]
        image_path = tf.keras.utils.get_file(name, origin=url)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        return img

    # Normalize an image
    def deprocess(self, img):
        img = 255*(img + 1.0)/2.0
        return tf.cast(img, tf.uint8)


    # Display an image
    def show(self, img):
        plt.figure(figsize=(12,12))
        plt.grid(False)
        plt.axis('off')
        plt.imshow(img)
        plt.show()

        # Downsizing the image makes it easier to work with.
        original_img = self.download(self.url, target_size=[225, 375])
        original_img = np.array(original_img)
        self.original_img = original_img
        self.show(original_img)
