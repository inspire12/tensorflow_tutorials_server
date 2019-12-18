from __future__ import absolute_import, division, print_function, unicode_literals

from mimetypes import guess_type

import tensorflow as tf
import tensorflow_hub as hub

import os
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import PIL.Image

''' 
keras
https://www.tensorflow.org/tutorials/generative/style_transfer
'''
class GanService:
    def __init__(self):
        mpl.rcParams['figure.figsize'] = (12,12)
        mpl.rcParams['axes.grid'] = False
        self.root_path = os.path.abspath("./")
        self.transfer_url = os.path.join(self.root_path, 'resource', 'tmp_img', "transfer.png")
        pass

    def run(self, content_name='target1.png', style_name='styles.jpg'):
        content_path = os.path.join(self.root_path, 'resource', 'img', content_name)
        # print(content_path)
        if not os.path.exists(content_path):
            content_path  = tf.keras.utils.get_file(os.path.join(self.root_path, 'resource', 'tmp_img', 'target.png'), content_name)
        style_path = os.path.join(self.root_path, 'resource', 'img', style_name)
        # print(style_path)
        if not os.path.exists(style_path):
            style_path = tf.keras.utils.get_file(os.path.join(self.root_path, 'resource', 'tmp_img', 'style.png'), style_name)

        return self.__transfer_image(content_path, style_path)

    def __transfer_image(self, content_path, style_path):
        content_image = self.load_img(content_path)
        style_image = self.load_img(style_path)

        # plt.subplot(1, 2, 1)
        # self.imshow(content_image, 'Content Image')
        #
        # plt.subplot(1, 2, 2)
        # self.imshow(style_image, 'Style Image')
        # plt.show()

        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
        stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
        transfer_img = self.tensor_to_image(stylized_image)

        transfer_img.save(self.transfer_url)
        # transfer_img.show()
        return self.transfer_url

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

    def transferred_path(self):
        return self.transfer_url

    def get_content_type(self):
        content_type, _ = guess_type(self.transfer_url)
        return content_type
