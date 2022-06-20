import os
import zipfile

"""
Este módulo establece las variables globales del proyecto y crea los directorios necesarios 
si no existen
"""

def initialize():
    global PUNCT_MARK_DICT      # diccionario con el mapeo de los signos de puntuación y tokens. Usado en el apartado 6
    global PUNCT_MARKS          # lista con signos de puntuación ['.',',',';',':','?','!']
    global MAYUS_MARKS          # lista de signos de puntuación de final de oración ['.','?','!']
    global PUNCT_MARKS_TOKENS   # lista de tokens de los signos de puntuación [".PERIOD",",COMMA",";SEMICOLON",":COLON","?QUESTIONMARK","!EXCLAMATIONMARK"]
    global MAYUS_MARKS_TOKENS   # lista de tokens correspondientes a signos de puentuación de final de oración
    global NUM                  # token de números <NUM>
    global DASH                 # token de guion -DASH

    global CURRENT_DIR              # ruta a directorio raíz
    global DATA_DIR                 # ruta a directorio de datos
    global DATA_RAW_DIR             # ruta a directorio de archivos del dataset 
    global DATA_PREPROCESSED_DIR    # ruta a directorio de archivos de datos preprocesados (con signos tokenizados). Usado en apartado 6
    global DATA_PREPARED_DIR        # ruta a directorio de archivos procesados para poder ser pasados al modelo de Ottokar Tilk y Tanel Alum (Apartado 6)

    global PUNCTUATOR2TF2_DIR      # ruta al directorio de archivos del modelo de Ottokar Tilk y Tanel Alum (Apartado 6)
    global PREDICTED_DIR            # ruta al directorio de archivos con predicciones de modelos

    """
    Estructura de directorios:

    -root
        |-data
        |  |-prepared
        |  |-preprocessed
        |  |-raw
        |-predicted
    """

    PUNCT_MARK_DICT = {".": ".PERIOD", ",": ",COMMA", ";": ";SEMICOLON", ":": ":COLON", "?": "?QUESTIONMARK", "!": "!EXCLAMATIONMARK"}
    PUNCT_MARKS = ['.',',',';',':','?','!'] # Signos de puntuación (sdp)
    MAYUS_MARKS = ['.','?','!'] # Signos de final de oración
    PUNCT_MARKS_TOKENS = [PUNCT_MARK_DICT[p] for p in PUNCT_MARKS]
    MAYUS_MARKS_TOKENS = [PUNCT_MARK_DICT[p] for p in MAYUS_MARKS]
    NUM = '<NUM>'
    DASH = '-DASH'

    CURRENT_DIR = os.path.curdir
    DATA_DIR = os.path.join(CURRENT_DIR,'data')
    DATA_RAW_DIR = os.path.join(DATA_DIR,'raw')
    DATA_PREPROCESSED_DIR = os.path.join(DATA_DIR,'preprocessed')
    DATA_PREPARED_DIR = os.path.join(DATA_DIR,'prepared')

    PUNCTUATOR2TF2_DIR = os.path.join(CURRENT_DIR,'punctuator2tf2')
    PREDICTED_DIR = os.path.join(CURRENT_DIR,'predicted')

    # zips con el dataset y el modelo de Ottokar Tilk y Tanel Alum en caso necesario
    dataset_zip_dir = os.path.join(CURRENT_DIR,'zips','PLN-MULCIA-Junio-2022-Dataset.zip')
    punctuator2tf2_zip_dir = os.path.join(CURRENT_DIR,'zips','punctuator2tf2.zip')
    
    # creación de directorios y descompresión de archivos en caso necesario
    if not os.path.exists(DATA_DIR):
        print('Data directories generated')
        os.makedirs(DATA_DIR)
    if not os.path.exists(DATA_RAW_DIR):
        os.makedirs(DATA_RAW_DIR)
        try:
            dataset_zip = zipfile.ZipFile(dataset_zip_dir,'r')
            dataset_zip.extractall(pwd=None, path=DATA_RAW_DIR)
            print('Dataset extracted in '+DATA_RAW_DIR)
            dataset_zip.close()
        except:
            pass
    elif len(os.listdir(DATA_RAW_DIR)) == 0:
        try:
            dataset_zip = zipfile.ZipFile(dataset_zip_dir,'r')
            dataset_zip.extractall(pwd=None, path=DATA_RAW_DIR)
            print('Dataset extracted in '+DATA_RAW_DIR)
            dataset_zip.close()
        except:
            pass
    if not os.path.exists(DATA_PREPARED_DIR):
        os.makedirs(DATA_PREPARED_DIR)
    if not os.path.exists(DATA_PREPROCESSED_DIR):
        os.makedirs(DATA_PREPROCESSED_DIR)
    if not os.path.exists(PREDICTED_DIR):
        os.makedirs(PREDICTED_DIR)

    if not os.path.exists(PUNCTUATOR2TF2_DIR):
        os.makedirs(PUNCTUATOR2TF2_DIR)
        try:
            punctuator2tf2_zip = zipfile.ZipFile(punctuator2tf2_zip_dir,'r')
            punctuator2tf2_zip.extractall(pwd=None, path=PUNCTUATOR2TF2_DIR)
            print('Model punctuator2tf2 extracted in '+PUNCTUATOR2TF2_DIR)
            punctuator2tf2_zip.close()
        except:
            pass