from __future__ import absolute_import, division, print_function, unicode_literals

import io

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


class WordService:
    def __init__(self):
        tfds.disable_progress_bar()
        pass

    def run(self):
        embedding_layer = keras.layers.Embedding(1000, 5)
        result = embedding_layer(tf.constant([1,2,3]))
        result.numpy()
        result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))
        result.shape
        (train_data, test_data), info = tfds.load(
            'imdb_reviews/subwords8k',
            split = (tfds.Split.TRAIN, tfds.Split.TEST),
            with_info=True, as_supervised=True)
        encoder = info.features['text'].encoder
        encoder.subwords[:20]
        padded_shapes = ([None],())

        train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)
        test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)

        train_batch, train_labels = next(iter(train_batches))
        train_batch.numpy()

        embedding_dim=16
        model = self.__load_model(encoder, embedding_dim)

        history = model.fit(
            train_batches,
            epochs=10,
            validation_data=test_batches, validation_steps=20)

        return self.__show_plt(history)


    def __load_model(self, encoder, embedding_dim):
        model = keras.Sequential([
            keras.layers.Embedding(encoder.vocab_size, embedding_dim),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.summary()
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def __show_plt(self, history):
        history_dict = history.history

        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12,9))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12,9))
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.ylim((0.5,1))
        plt.show()

        memdata = io.BytesIO()
        plt.savefig(memdata, format='png')
        image = memdata.getvalue()
        return image