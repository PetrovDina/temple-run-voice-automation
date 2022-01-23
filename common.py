import tensorflow.keras as keras

# ----------------------- CHOOSE MODE ------------------------------
MODE = "SRB"  # "SRB" for serbian language, "ENG" for english language
# --------------------------------------------------------------------

if MODE == "SRB":
    COMMANDS = ['skoci', 'dole', 'levo', 'desno']
    DATASET_PATH = 'srb-sgram\\{}\\*.png'
    MODEL = keras.models.load_model('models\\model-srb-30epochs.h5')

elif MODE == "ENG":
    COMMANDS = ['up', 'down', 'left', 'right']
    DATASET_PATH = 'speech-commands-sgram\\{}\\*.png'
    MODEL = keras.models.load_model('models\\model-eng-20epochs.h5')

