# coding: utf-8
from __future__ import division

import random
import os
import sys
import operator
import pickle
import codecs
import fnmatch

"""
Archivo adaptado para el Trabajo de Predicción de Puntuación de la asignatura Procecsamiento del Lenguaje Natural del Máster MULCIA
    Repositorio original: https://github.com/ottokart/punctuator2
    Reimplementación de los autores en tensorflor : https://github.com/cadia-lvl/punctuation-prediction/tree/master/punctuator2tf2
"""

DATA_PATH = os.path.join('data','preprocessed')     # directorio donde se almacenan los corpus preparados para el modelo
END = "</S>"                                        # token de final de oración
UNK = "<UNK>"                                       # token de términos desconocidos
SPACE = "_SPACE"                                    # token de espacio
MAX_WORD_VOCABULARY_SIZE = 100000                   # tamaño máximo del vocabulario
MIN_WORD_COUNT_IN_VOCAB = 2                         # tamaño mínimo de vocabulario
MAX_SEQUENCE_LEN = 200                              # tamaño máximo de una oración procesada.

# Las siguientes rutas corresponden al split del 
# archivo original de test preparadas para el modelo
TRAIN_FILE = os.path.join(DATA_PATH, "train")               # ruta del archivo de entrenamiento preparado
DEV_FILE = os.path.join(DATA_PATH, "dev")                   # ruta del archivo dev preparado
TEST_FILE = os.path.join(DATA_PATH, "test")                 # ruta del archivo test preparado

WORD_VOCAB_FILE = os.path.join(DATA_PATH, "vocabulary")     # ruta al archivo de vocabulario de entrenamiento
PUNCT_VOCAB_FILE = os.path.join(DATA_PATH, "punctuations")  # ruta al archivo con signos ded puntuación considerados en el entrenamiento

# Signos de puntuación considerados en el entrenamiento
PUNCTUATION_VOCABULARY = {SPACE, ",COMMA", ".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK", ":COLON", ";SEMICOLON"}
# Sustitución (mapping) de signos en caso de que se quieran sustituir
PUNCTUATION_MAPPING = {}

# tokens de final de oración
EOS_TOKENS = {".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK"}

# tokens descartados
CRAP_TOKENS = {
    "<doc>",
    "<doc.>",
} 
PAUSE_PREFIX = "<sil="


def add_counts(word_counts, line):
    """
    Realiza el conteo (ocurrencias) de cada palabra de una oracion que no sea un token de
    puntuacion o descartado

    input:
        word_counts: diccionario con claves las palabras y valores sus ocurrencias.
        line: string a procesar.
    
    output:
        Diccionario word_counts actualizado.
    """

    # Si es un token de puntuación o descartado, se ignora
    for w in line.split():
        if (
            w in CRAP_TOKENS
            or w in PUNCTUATION_VOCABULARY
            or w in PUNCTUATION_MAPPING
            or w.startswith(PAUSE_PREFIX)
        ):
            continue
        # En caso contrario se actualiza el diccionario
        word_counts[w] = word_counts.get(w, 0) + 1


def create_vocabulary(word_counts):
    """
    Genera la lista de palabras presentes en el corpus (vocabulario)

    input:
        word_counts: diccionario con claves las palabras y valores sus ocurrencias.
    output:
        Lista de vocabulario.
    """
    vocabulary = [
        wc[0]
        for wc in reversed(sorted(word_counts.items(), key=operator.itemgetter(1)))
        if wc[1] >= MIN_WORD_COUNT_IN_VOCAB and wc[0] != UNK
    ][
        :MAX_WORD_VOCABULARY_SIZE
    ]  # Unk will be appended to end

    vocabulary.append(END)
    vocabulary.append(UNK)

    print("Vocabulary size: %d" % len(vocabulary))
    return vocabulary


def iterable_to_dict(arr):
    """
    Función auxiliar que genera un diccionario enumerado a partir de un objeto iterable
    """
    return dict((x.strip(), i) for (i, x) in enumerate(arr))


def read_vocabulary(file_name):
    """
    Función auxiliar que devuele un diccionario enumerado a partir del archivo de vocabulario
    """
    with codecs.open(file_name, "r", "utf-8") as f:
        vocabulary = f.readlines()
        print('Vocabulary "%s" size: %d' % (file_name, len(vocabulary)))
        return iterable_to_dict(vocabulary)


def write_vocabulary(vocabulary, file_name):
    """
    Función auxiliar que escribe en el archivo de vocabulario
    """
    with codecs.open(file_name, "w", "utf-8") as f:
        f.write("\n".join(vocabulary))


def write_processed_dataset(input_files, output_file):
    """
    Genera las vectorizaciones a partir del vocabulario generado  y crea un archivo a partir  de los
    archivos originales preprocesados.
    """

    # lista de vectorizaciones
    data = []

    # lectura del vocabulario
    word_vocabulary = read_vocabulary(WORD_VOCAB_FILE)
    # lectura del vocabulario de puntuaciones consideradas
    punctuation_vocabulary = read_vocabulary(PUNCT_VOCAB_FILE)

    num_total = 0   # cantidad de tokens procesados
    num_unks = 0    # cantidad de tokens desconocidos procesados

    current_words = []          # lista con vectorizaciones de palabras
    current_punctuations = []   # lista con vectorizaciones de signos de puntuación
    current_pauses = []         # lista con vectorizaciones de pausas (si procede)

    last_eos_idx = 0                    # si sigue siendo 0 cuando se alcanza MAX_SEQUENCE_LEN, entonces la frase es demasiado larga y se salta.
    last_token_was_punctuation = True   # omitir el primer token si es un signo de puntuación
    last_pause = 0.0

    skip_until_eos = False  # if a sentence does not fit into subsequence, then we need to skip tokens until we find a new sentence


    # iteramos en los archivos de entrada
    for input_file in input_files:
        # Abrimos los archivos
        with codecs.open(input_file, "r", "utf-8") as text:
            # iteramos en las líneas del archivo
            for line in text:
                # iteramos en los tokens de la linea
                for token in line.split():

                    # Mapeo (sustitución de token) si procede
                    if token in PUNCTUATION_MAPPING:
                        token = PUNCTUATION_MAPPING[token]

                    # Saltar de linea si no se empieza la oración hasta símbolo EOS
                    if skip_until_eos:

                        if token in EOS_TOKENS:
                            skip_until_eos = False

                        continue
                    # descartar token si procede
                    elif token in CRAP_TOKENS:
                        continue
                    
                    # sustitución de prefijo de pausa si procede
                    elif token.startswith(PAUSE_PREFIX):
                        last_pause = float(
                            token.replace(PAUSE_PREFIX, "").replace(">", "")
                        )

                    # si el token es un signo de puntuación
                    elif token in punctuation_vocabulary:
                        # quedarse con el primero si aparecen varios consecutivos
                        if (
                            last_token_was_punctuation
                        ):  # if we encounter sequences like: "... !EXLAMATIONMARK .PERIOD ...", then we only use the first punctuation and skip the ones that follow
                            continue
                        
                        # si el token es EOS se actualiza last_eos_idx
                        if token in EOS_TOKENS:
                            last_eos_idx = len(
                                current_punctuations
                            )  # no -1, because the token is not added yet

                        punctuation = punctuation_vocabulary[token]

                        # se añade a la lista de puntuaciones el token de puntuación correspondiente
                        current_punctuations.append(punctuation)
                        last_token_was_punctuation = True
                    
                    # en otro caso es una palabra
                    else:
                        # si no era de puntuación, añadir un espacio
                        if not last_token_was_punctuation:
                            current_punctuations.append(punctuation_vocabulary[SPACE])

                        # busca la palabra en el vocabulario, en caso contrario es desconocida
                        word = word_vocabulary.get(token, word_vocabulary[UNK])

                        # Añade a las listas
                        current_words.append(word)
                        current_pauses.append(last_pause)
                        last_token_was_punctuation = False

                        num_total += 1
                        num_unks += int(word == word_vocabulary[UNK])

                    # si se alcanza el final de la secuencia
                    if (
                        len(current_words) == MAX_SEQUENCE_LEN
                    ):  # this also means, that last token was a word

                        assert len(current_words) == len(current_punctuations) + 1, (
                            "#words: %d; #punctuations: %d"
                            % (len(current_words), len(current_punctuations))
                        )
                        assert current_pauses == [] or len(current_words) == len(
                            current_pauses
                        ), (
                            "#words: %d; #pauses: %d"
                            % (len(current_words), len(current_pauses))
                        )

                        # descarta oración si es demasiado larga y salta de línea
                        if last_eos_idx == 0:
                            skip_until_eos = True

                            current_words = []
                            current_punctuations = []
                            current_pauses = []

                            last_token_was_punctuation = True  # next sequence starts with a new sentence, so is preceded by eos which is punctuation

                        # en caso contrario se añade a la lista de vectorizaciones
                        else:
                            subsequence = [
                                current_words[:-1] + [word_vocabulary[END]],
                                current_punctuations,
                                current_pauses[1:],
                            ]

                            data.append(subsequence)

                            # Carry unfinished sentence to next subsequence
                            current_words = current_words[last_eos_idx + 1 :]
                            current_punctuations = current_punctuations[
                                last_eos_idx + 1 :
                            ]
                            current_pauses = current_pauses[last_eos_idx + 1 :]

                        last_eos_idx = 0  # sequence always starts with a new sentence

    print("%.2f%% UNK-s in %s" % (num_unks / num_total * 100, output_file))

    # se guardan las verctorizaciones en un archivo codificado
    with open(output_file, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def create_dev_test_train_split_and_vocabulary(
    root_path, build_vocabulary, train_output, dev_output, test_output
):
    """
    Función que crea los archivos necesarios de train, dev y test
    realmente test no es necesario pero lo dejamos para mantener lo máximo posible el código 
    original
    """

    train_txt_files = []
    dev_txt_files = []
    test_txt_files = []

    if build_vocabulary:
        word_counts = dict()

    for root, _, filenames in os.walk(root_path):
        for filename in fnmatch.filter(filenames, "*.txt"):

            path = os.path.join(root, filename)

            if filename.endswith(".test.txt"):
                test_txt_files.append(path)

            elif filename.endswith(".dev.txt"):
                dev_txt_files.append(path)

            else:
                train_txt_files.append(path)

                if build_vocabulary:
                    with codecs.open(path, "r", "utf-8") as text:
                        for line in text:
                            add_counts(word_counts, line)

    if build_vocabulary:
        vocabulary = create_vocabulary(word_counts)
        write_vocabulary(vocabulary, WORD_VOCAB_FILE)
        punctuation_vocabulary = iterable_to_dict(PUNCTUATION_VOCABULARY)
        write_vocabulary(punctuation_vocabulary, PUNCT_VOCAB_FILE)

    write_processed_dataset(train_txt_files, train_output)
    write_processed_dataset(dev_txt_files, dev_output)
    write_processed_dataset(test_txt_files, test_output)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        sys.exit(
            "The path to the source data directory with txt files is missing. The command should be: python data.py {folder with train, test and dev splits}"
        )

    if len(os.listdir(DATA_PATH)) != 0:
        sys.exit("Data already exists")

    create_dev_test_train_split_and_vocabulary(
        path, True, TRAIN_FILE, DEV_FILE, TEST_FILE
    )

