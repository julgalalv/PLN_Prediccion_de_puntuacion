import os
def initialize():
    global PUNCT_MARK_DICT
    global PUNCT_MARKS
    global MAYUS_MARKS
    global PUNCT_MARKS_TOKENS
    global MAYUS_MARKS_TOKENS
    global NUM
    global DASH

    global CURRENT_DIR
    global DATA_DIR
    global DATA_RAW_DIR
    global DATA_PREPROCESSED_DIR
    global DATA_PREPARED_DIR

    PUNCT_MARK_DICT = {".": ".PERIOD", ",": ",COMMA", ";": ";SEMICOLON", ":": ":COLON", "?": "?QUESTIONMARK", "!": "!EXCLAMATIONMARK"}
    PUNCT_MARKS = ['.',',',';',':','?','!'] # Signos de puntuación (sdp)
    MAYUS_MARKS = ['.','?','!'] # Signos que afectan a las mayúscualas del entorno
    PUNCT_MARKS_TOKENS = [PUNCT_MARK_DICT[p] for p in PUNCT_MARKS]
    MAYUS_MARKS_TOKENS = [PUNCT_MARK_DICT[p] for p in MAYUS_MARKS]
    NUM = '<NUM>'
    DASH = '-DASH'

    CURRENT_DIR = os.path.curdir
    DATA_DIR = os.path.join(CURRENT_DIR,'data')
    DATA_RAW_DIR = os.path.join(DATA_DIR,'raw')
    DATA_PREPROCESSED_DIR = os.path.join(DATA_DIR,'preprocessed')
    DATA_PREPARED_DIR = os.path.join(DATA_DIR,'prepared')

