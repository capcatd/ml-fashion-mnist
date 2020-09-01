import os
import random

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

weights_file = 'weights.h5'
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def draw_image(i):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(test_images[i], cmap=plt.cm.binary)

    if np.argmax(predictions[i]) == test_labels[i]:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel(
        "id:{}\npredicted: {} with {:2.0f}% accuracy\noriginal: {}".format(i, class_names[np.argmax(predictions[i])],
                                                                           100 * np.max(predictions[i]),
                                                                           class_names[test_labels[i]]),
        color=color)


def load_data() -> (np.array, np.array, np.array, np.array):
    # get data from server
    (train_im, train_l), (test_im, test_l) = keras.datasets.fashion_mnist.load_data()

    return train_im / 255.0, train_l, test_im / 255.0, test_l


def create_model() -> keras.Sequential:
    # create layers
    m = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])

    # compile
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return m


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_data()
    model = create_model()

    # check if exist weights.h5
    if os.path.exists(weights_file):
        # load weights
        model.load_weights(weights_file)
    else:
        # save weights
        callback = keras.callbacks.ModelCheckpoint(weights_file,
                                                   monitor='accuracy',
                                                   mode='max',
                                                   save_best_only=True)
        # train model
        model.fit(train_images, train_labels, epochs=8, callbacks=[callback])

    # test loss and accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nТочность на проверочных данных:', test_acc)

    predictions = model.predict(test_images)

    count_x = 5
    count_y = 5

    if 0 < count_y * count_x < 10000:
        print('drawing graph ...')
        plt.figure(figsize=(5 * count_x, 5 * count_y))
        for order, i in enumerate(random.sample(range(0, 10000), count_x * count_y)):
            plt.subplot(count_x, count_y, order + 1)
            draw_image(i)

        plt.show()
    else:
        print('invalid count_x or count_y')
