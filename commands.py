import tensorflow.keras as keras


COMMANDS_ENG = ['up', 'down', 'left', 'right']
COMMANDS_SRB = ['skoci', 'dole', 'levo', 'desno']
COMMANDS = COMMANDS_SRB

DATASET_PATH_ENG = 'speech-commands-sgram\\{}\\*.png'
DATASET_PATH_SRB = 'srb-sgram\\{}\\*.png'

DATASET_PATH_32_X_32 = DATASET_PATH_SRB
DATASET_PATH_124_X_124 = 'speech-commands-sgram-124x124\\{}\\*.png'

MODEL = keras.models.load_model('model-32x32-20epoch.h5')