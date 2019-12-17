from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import IPython.display as display
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import time
import functools

class GenService:
    def __init__(self):
        pass

    def run(self):
        root_path = os.path.abspath("./")
        print(root_path)
        content_path = tf.keras.utils.get_file(os.path.normpath(root_path +'/resource/img/test.jpg'), 'https://post-phinf.pstatic.net/MjAxOTA5MzBfNCAg/MDAxNTY5ODE5NzI0NDU5.R26WW6glEa07UkqxvFDGBPPoLoHFIpowlZRBzfYQnzYg.04W2nSQ7LWyCCh6FrYItmpQ4d5uv_hyk1xF53pIc2xIg.JPEG/%EC%97%94%EC%94%A8%EC%86%8C%ED%94%84%ED%8A%B8_%EC%97%94%EC%94%A8%EC%86%8C%ED%94%84%ED%8A%B8%2C_2019%EB%85%84_%EC%8B%A0%EC%9E%85%EC%82%AC%EC%9B%90_%EA%B3%B5%EA%B0%9C%EB%AA%A8%EC%A7%91_%EC%8B%A4%EC%8B%9C.jpg?type=w1200')

        # https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
        style_path = tf.keras.utils.get_file('real_person.jpg','data:image/jpeg;base64,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')
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
