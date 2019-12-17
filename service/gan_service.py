from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_hub as hub

import os
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import PIL.Image


class GanService:
    def __init__(self):
        mpl.rcParams['figure.figsize'] = (12,12)
        mpl.rcParams['axes.grid'] = False
        pass

    def run(self):
        root_path = os.path.abspath("./")
        print(root_path)
        # https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
        style_path = os.path.join(root_path, 'resource', 'img', 'styles.jpg')

        content_path = os.path.join(root_path, 'resource', 'img', 'target1.png')
        print(content_path)
        content_image = self.load_img(content_path)
        style_image = self.load_img(style_path)

        plt.subplot(1, 2, 1)
        self.imshow(content_image, 'Content Image')

        plt.subplot(1, 2, 2)
        self.imshow(style_image, 'Style Image')
        plt.show()

        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
        stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
        transfer_img = self.tensor_to_image(stylized_image)

        transfer_img.save("transfer.png")
        transfer_img.show()


    def imshow(self, image, title=None):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)

        plt.imshow(image)
        if title:
            plt.title(title)

    def load_img(self, path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def tensor_to_image(self, tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)
