# coding: utf-8
from __future__ import division

import models, data, main

import sys
import codecs

import tensorflow as tf
import numpy as np

"""
Archivo adaptado para el Trabajo de Predicción de Puntuación de la asignatura Procecsamiento del Lenguaje Natural del Máster MULCIA
    Repositorio original: https://github.com/ottokart/punctuator2
    Reimplementación de los autores en tensorflor : https://github.com/cadia-lvl/punctuation-prediction/tree/master/punctuator2tf2

    Hemos modificado el puntuador original, que no capitalizaba las palabras tras signos de final de oración, para que lo haga.
    Además hemos añadido la puntuación básica para poner en mayúscula la primera letra de la oración y finalizar con .PERIOD
    si no se ha predicho un signo de final de oración.
"""

MAX_SUBSEQUENCE_LEN = 200           # tamaño máximo de secuenci a procesar

def to_array(arr, dtype=np.int32):
    """
    Función auxiliar que transforma a array de numpy
    """
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T

def convert_punctuation_to_readable(punct_token):
    """
    Función auxiliar que transforma un token de puntuación en el signo correspondiente.
    """
    if punct_token == data.SPACE:
        return " "
    else:
        return punct_token[0]

def to_upper(text):
    """
    Por Julián: Función auxiliar que pone en mayúscula la primera letra de text
    """
    return text[0].upper() + (text[1:] if len(text) > 1 else '')

def restore(text, word_vocabulary, reverse_punctuation_vocabulary, model):
    """
    Función princial que devuelve el string text puntuado
    """

    i = 0               # contador de elemento en la secuencia text
    punctuated = ''     # string puntuado inicializado a vacío
    while True:
        subsequence = text[i:i+MAX_SUBSEQUENCE_LEN]     # subsecuencia comprendida entre el último roken procesado y el tamaño máximo de subsecuencia
        # Si no hay subsecuencia se ha terminado
        if len(subsequence) == 0:
            break
        # procesado de la susecuencia para vectorizar las palabras
        converted_subsequence = [word_vocabulary.get(w, word_vocabulary[data.UNK]) for w in subsequence]
        # predicción por el modelo dada la subsecuencia
        y = predict(to_array(converted_subsequence), model)
        # se añade el primer token ya que para este no hay predicción
        punctuated += subsequence[0]
        last_eos_idx = 0
        punctuations = [] # signos de puntuación procesados
        # para cada token de la predicción 
        for y_t in y:
            p_i = np.argmax(tf.reshape(y_t, [-1]))  # el signo final es aquel que maximiza la probabilidad devuelta por la predicción
            punctuation = reverse_punctuation_vocabulary[p_i] # se obtiene el token de la predicción
            punctuations.append(punctuation)
            if punctuation in data.EOS_TOKENS:
                last_eos_idx = len(punctuations) # we intentionally want the index of next element
        # definición del número de elementos a procesar en la secuencia predicha
        if subsequence[-1] == data.END:
            step = len(subsequence) - 1
        elif last_eos_idx != 0:
            step = last_eos_idx
        else:
            step = len(subsequence) - 1
        # iteramos en los elementos de la secuencia predicha
        for j in range(step):
            # si no es el final
            if j < step - 1:
                # añadimos el signo de puntuación correspondiente si no es un espacio
                punctuated += " " + punctuations[j] + " " if punctuations[j] != data.SPACE else " "
                # AÑADIDO POR JULIÁN: si el signo añadido era EOS, ponemos en mayúsculas la siguiente palabra.
                punctuated += subsequence[1+j] if not punctuations[j] in data.EOS_TOKENS  else to_upper(subsequence[1+j])
            # AÑADIDO POR JULIÁN: si el último token no es EOS se añade el token de punto (PUNTUACIÓN BÁSICA)
            elif j == step -1: punctuated += " .PERIOD " if punctuations[j] not in data.EOS_TOKENS else  " " + punctuations[j] + " "
        if subsequence[-1] == data.END:
            break
        i += step
    # AÑADIDO POR JULIÁN: Se pone en mayúsculas la primera letra (PUNTUACIÓN BÁSICA)
    return to_upper(punctuated)



def predict(x, model):
    """
    Devuelve las probabilidades de cda token a partir de un softmax de la salida del modelo.
    """
    return tf.nn.softmax(net(x))

if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    else:
        sys.exit("Model file path argument missing")

    if len(sys.argv) > 2:
        input_file = sys.argv[2]
    else:
        sys.exit("Input file path argument missing")

    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    else:
        sys.exit("Output file path argument missing")

    vocab_len = len(data.read_vocabulary(data.WORD_VOCAB_FILE))
    x_len = vocab_len if vocab_len < data.MAX_WORD_VOCABULARY_SIZE else data.MAX_WORD_VOCABULARY_SIZE + data.MIN_WORD_COUNT_IN_VOCAB
    x = np.ones((x_len, main.MINIBATCH_SIZE)).astype(int)

    print("Loading model parameters...")
    net, _ = models.load(model_file, x)

    print("Building model...")

    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary

    reverse_word_vocabulary = {v:k for k,v in word_vocabulary.items()}
    reverse_punctuation_vocabulary = {v:k for k,v in punctuation_vocabulary.items()}

    # Se abre el archivo a puntuar
    with codecs.open(input_file, 'r', 'utf-8') as in_file,  open(output_file, 'w') as out_file:
        lines = in_file.readlines()

        if len(lines) == 0:
            sys.exit("Input file empty.")

        n_line = 0
        # se procesa línea a línea y se puntúa con restore
        for line in lines:
            if (n_line % (len(lines) // 10 +1)) == 0: print('Processing line: {}/{}'.format(n_line,len(lines)))
            input_text = line.strip()
            text = [w for w in input_text.split() if w not in punctuation_vocabulary and w not in data.PUNCTUATION_MAPPING and not w.startswith(data.PAUSE_PREFIX)] + [data.END]
            punct = restore(text, word_vocabulary, reverse_punctuation_vocabulary, net)
            out_file.write(punct+'\n')
            n_line +=1
        print('Processing line: {}/{}'.format(len(lines),len(lines)))
        print('Process Finished')
    in_file.close()
    out_file.close()

