from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt


class TestRun:
    def __init__(self):
        print(tf.__version__)

    def test(self):
        tf.debugging.set_log_device_placement(True)

        # 텐서 생성
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print(c)

    def run_mnist(self):
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0


    def run_fashion_mnist(self):
        # https://github.com/tensorflow/docs/blob/master/site/ko/tutorials/keras/classification.ipynb
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        print(train_images.shape)
        print(len(train_labels))
        # plt.figure()
        # plt.imshow(train_images[0])
        # plt.colorbar()
        # plt.grid(False)
        # plt.show()

        train_images = train_images / 255.0
        test_images = test_images / 255.0

        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[i]])
        plt.show()

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        # 모델 컴파일

        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

        # 모델 훈련
        # 손실 함수(Loss function)
        #
        model.fit(train_images, train_labels, epochs=5)

        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

        print('\n테스트 정확도:', test_acc)
