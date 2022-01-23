import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image

from predict import plot_confusion_matrix
import pandas as pd


DATASET_PATH_32_X_32 = 'speech-commands-sgram\\{}\\*.png'
DATASET_PATH_124_X_124 = 'speech-commands-sgram-124x124\\{}\\*.png'

SPECTROGRAM_DIMENSIONS = 32


# Hyper-parameters
IMAGE_HEIGHT = SPECTROGRAM_DIMENSIONS
IMAGE_WIDTH = SPECTROGRAM_DIMENSIONS
BATCH_SIZE = 32
EPOCHS = 20

TEST_RATIO = 0.1
VALIDATION_RATIO = 0.2

COMMANDS = ['up', 'down', 'left', 'right']


def load_data():
    images = []
    labels = []

    for i in range(len(COMMANDS)):
        command = COMMANDS[i]

        for file_name in glob(DATASET_PATH_32_X_32.format(command)):
            image = np.array(Image.open(file_name))

            images.append(image)
            labels.append(i)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def create_model(input_shape):
    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(4, activation='softmax'))

    return model


def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["acc"], label="train accuracy")
    axs[0].plot(history.history["val_acc"], label="validation accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="validation error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    # load data
    images, labels = load_data()
    images, labels = shuffle(images, labels, random_state=42)

    # create train, validation and test split
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size)
    train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels,
                                                                                        test_size=validation_size)

    return train_images, validation_images, test_images, train_labels, validation_labels, test_labels


def training():
    # get train, validation, test splits
    train_images, validation_images, test_images, train_labels, validation_labels, test_labels = prepare_datasets(TEST_RATIO, VALIDATION_RATIO)

    # create network
    input_shape = (IMAGE_HEIGHT, IMAGE_HEIGHT, 4)
    model = create_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    model.summary()

    # train model
    history = model.fit(train_images, train_labels, validation_data=(validation_images, validation_labels),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS)

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    # plot confusion matrix
    prediction = model.predict(test_images)
    prediction = np.argmax(prediction, axis=1)
    plot_confusion_matrix(test_labels, prediction)

    # save history
    hist_df = pd.DataFrame(history.history)
    with open('hist_json_file.json', mode='w') as f:
        hist_df.to_json(f)

    model.save('test-model-32x32-30-epochs.h5')


if __name__ == '__main__':
    print('Temple Run - Voice Automation Training')
    training()
