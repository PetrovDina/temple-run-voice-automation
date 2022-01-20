import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from glob import glob
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
import seaborn as sns


commands = ['up', 'down', 'left', 'right']


def load_data():
    X = []
    y = []

    for i in range(len(commands)):
        command = commands[i]

        for file_name in glob('speech-commands-sgram\\{}\\*.png'.format(command)):
            image = np.array(Image.open(file_name))

            print(file_name)
            print(image.shape)
            X.append(image)
            y.append(i)

    X = np.array(X)
    y = np.array(y)
    return X, y


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
    axs[0].plot(history.history["val_acc"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


# TODO call this method after prediction on test set
def plot_confusion_matrix(y_test, prediction):
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    sns.heatmap(confusion_matrix(y_true=y_test, y_pred=prediction), ax=ax, xticklabels=commands, yticklabels=commands,
                annot=True,
                alpha=0.7, linewidths=2)
    fig.text(s='Confusion Matrix', size=20, fontweight='bold',
             fontname='monospace', y=0.92, x=0.28, alpha=0.8)

    plt.show()


def prepare_datasets(test_size, validation_size):

    # load data
    X, y = load_data()
    X, y = shuffle(X, y, random_state=42)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def training():
    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.1, 0.2)

    # create network
    input_shape = (32, 32, 4)
    model = create_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=4)

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    model.save('test-1.h5')


if __name__ == '__main__':
    print('Temple Run - Voice Automation')

    training()