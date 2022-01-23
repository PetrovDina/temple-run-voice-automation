import tensorflow.keras as keras

# ----------------------- CHOOSE MODE ------------------------------
MODE = "SRB"  # "SRB" for serbian language, "ENG" for english language
# --------------------------------------------------------------------

if MODE == "SRB":
    COMMANDS = ['skoci', 'dole', 'levo', 'desno']
    DATASET_PATH_32_X_32 = 'srb-sgram\\{}\\*.png'
    MODEL = keras.models.load_model('model-srb-32x32-30epochs.h5')

elif MODE == "ENG":
    COMMANDS = ['up', 'down', 'left', 'right']
    DATASET_PATH_32_X_32 = 'speech-commands-sgram\\{}\\*.png'
    MODEL = keras.models.load_model('model-eng-32x32-20epochs.h5')

