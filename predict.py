import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from commands import COMMANDS, MODEL


def plot_confusion_matrix(y_test, prediction):
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    sns.heatmap(confusion_matrix(y_true=y_test, y_pred=prediction), ax=ax, xticklabels=COMMANDS, yticklabels=COMMANDS,
                annot=True,
                alpha=0.7, linewidths=2, fmt='d')
    fig.text(s='Confusion Matrix', size=20, fontweight='bold',
             fontname='monospace', y=0.92, x=0.28, alpha=0.8)

    plt.show()


def predict_voice_input(audio_array):
    prediction = MODEL.predict(audio_array)
    return np.argmax(prediction, axis=1)[0], np.max(prediction, axis=1)[0]

'''
def predict_test_set():
    # get train, validation, test splits
    train_images, validation_images, test_images, train_labels, validation_labels, test_labels = prepare_datasets(TEST_RATIO, VALIDATION_RATIO)

    prediction = model.predict(test_images)
    prediction = np.argmax(prediction, axis=1)

    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plot_confusion_matrix(test_labels, prediction)


if __name__ == '__main__':
    print('Temple Run - Voice Automation - Prediction')
    predict_test_set()
'''