import os
import random

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

weights_file = 'weights.h5'
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
draw = False
EPOCHS = 4


def draw_comparison_graph(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']

    loss = history['loss']
    val_loss = history['val_loss']

    epochs_range = range(EPOCHS)

    print(epochs_range)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Точность на обучении')
    plt.plot(epochs_range, val_acc, label='Точность на валидации')
    plt.legend(loc='lower right')
    plt.title('Точность на обучающих и валидационных данных')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Потери на обучении')
    plt.plot(epochs_range, val_loss, label='Потери на валидации')
    plt.legend(loc='upper right')
    plt.title('Потери на обучающих и валидационных данных')
    # plt.savefig('./foo.png')
    plt.show()


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


def load_data() -> (np.array, np.array, np.array, np.array, np.array, np.array):
    # get data from server
    (train_im, train_l), (test_im, test_l) = keras.datasets.fashion_mnist.load_data()
    train_im = train_im / 255.0
    test_im = test_im / 255.0
    return train_im[:50000], train_l[:50000], train_im[50000:], train_l[50000:], test_im, test_l


def create_model() -> keras.Sequential:
    # create layers
    m = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])

    # compile
    m.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return m


if __name__ == '__main__':
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data()
    model = create_model()

    # check if exist weights.h5
    if False and os.path.exists(weights_file):
        # load weights
        model.load_weights(weights_file)
    else:
        # save weights
        callback = keras.callbacks.ModelCheckpoint(weights_file,
                                                   monitor='accuracy',
                                                   mode='max',
                                                   save_best_only=True)
        # train model
        history = model.fit(train_images, train_labels, epochs=EPOCHS, callbacks=[callback],
                            validation_data=(val_images, val_labels))

    # test loss and accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('history: ', history.history)
    print('\nТочность на проверочных данных:', test_acc)

    draw_comparison_graph(history.history)

    predictions = model.predict(test_images)

    if draw:
        count_x = 5
        count_y = 5

        if 0 < count_y * count_x < 10000:
            print('drawing graph ...')
            plt.figure(figsize=(3 * count_x, 3 * count_y))
            for order, i in enumerate(random.sample(range(0, 10000), count_x * count_y)):
                plt.subplot(count_x, count_y, order + 1)
                draw_image(i)

            plt.show()
        else:
            print('invalid count_x or count_y')
