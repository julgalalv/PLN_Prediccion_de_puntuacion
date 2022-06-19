import re
import os
import settings
import numpy as np

"""
En este módulo se definen funciones que son necesarias para el procesamiento de los datos. Pueden dividirse en:
    * APARTADOS 1 -5:
        * change_initial: Cambia la inicial de un string.
        * tokenizer: Devuelve la lista de tokens de un string.
        * ngrams: Devuelve la lista de Ngramas de un string.
        * padding: Dadas dos listas, añade el string '' a la de menor tamaño hasta que tenga 
                   el tamaño de la mayor.

    * APARTADO 6:
        * prepare_text: Preprocesa un texto para sustituir los tokens de puntuacion con respecto al 
                        mapeo definido en el trabajo de Ottokar Tilk y Tanel Alum.
        * prepare_file: Dado un archivo, usa prepare_text para preprocesarlo completamente.
        * train_dev_test_split: Separa un archivo de texto en conjuntos de train, dev y test. 
                                Esta separación la espera el modelo de Ottokar Tilk y Tanel Alum.

"""

###############################
### REFERENTES A APARTADOS 1-5:
###############################

def change_initial(string,uppercase):
    """
    Función que pone el primer caracter de un string en mayúscula o minúscula según 
    el parámetro booleano 'uppercase'

    input: 
        string: string a procesar
        uppercase: si True, pone la inicial a mayúscula, en caso contrario, a minúscula
    
    output: 
        string con la inicial cambiada
    """
    if not string:
        return
    init = string[0]
    new_initial = init.upper() if uppercase else init.lower()
    temp = list(string)
    temp[0] = new_initial
    return ''.join(temp)

def padding(list1, list2):
    """
    Dadas dos listas, añade '' a la menor hasta que ambas tengan el mismo tamaño.

    input:
        list1, list2: listas a procesar
    
    output:
        list1, list2 listas con padding. La de mayor tamaño no se modifica
    """
    len1, len2 = len(list1), len(list2)
    max_len = max(len1, len2)
    list1 = [*list1, *([''] * (max_len - len1))]
    list2 = [*list2, *([''] * (max_len - len2))]
    return list1, list2
    
def tokenizer(text,prepared = False):
    """
    Función que tokeniza un string

    input: 
        text: string a procesar
        prepared: si False, usa la expresión regular  r'?([.|,|;|:|!|?]+) ?' para añadir un espacio
                  a los signos de puntuación para poder ser separados mediante str.split(). En caso contrario el
                  texto ya viene preprocesado (APARTADO 6) y solo es necesario hacer el split(). 

    output: 
        lista de tokens de un string
    """

    if not prepared:
        # Generamos la expresión regular que añade espacios a los signos para poder separarlos con el método split()
        marks = ''.join(settings.PUNCT_MARKS)
        regex1, regex2 = r' ?([{}]+) ?'.format(marks), r' \1 '
        text = re.sub(regex1, regex2, text)
    return text.split()
 
def ngrams(text,N):
    """
    Devuelve la lista formada por las N-tuplas de cadenas de tamaño N presentes en el string 'text' (Ngramas)

    input: 
        text: string a procesar
        N: tamaño de las tuplas

    output: 
        lista de Ngramas de un string string
    """
    text = tokenizer(text)
    return [tuple(text[i:i+N]) for i in range(len(text)-N+1)]

############################
### REFERENTES A APARTADO 6:
############################

def prepare_text(text, lowercase=False, allow_duplicates=True,token_punct = True, token_num=True,token_dash = False):
    """
    Preprocesa un texto para sustituir los tokens de puntuacion con respecto al mapeo definido 
    en el trabajo de Ottokar Tilk y Tanel Alum. Esta transformación es necesaria pues el input del modelo
    espera dichos tokens en lugar de los signos de puntuación

    input:
        text: string a procesar
        lowercase: si True todo el texto se pone en minúsculas. En caso contrario se mantiene igual
        allow_duplicates: si True las cadenas de signos de puntuación duplicadas como los puntos suspensivos se mantienen
                          En caso contrario se sustituyen por una única aparición
        token_punct: si True se sustituyen los signos de puntuación por sus correspondientes tokens, definidos en 
                     settings.PUNCT_MARK_DICT
        token_punct: si True se sustituyen las cadenas numéricas aisladas (enteros o decimales con separación de coma
                     o punto que no aparecen concatenadas a caracteres alfabeticos o signos) por el token <NUM>
        token_dash: si True se sustituyen los guiones aislados (no concatenados a palabras o números como omega-3)
                    por el token -DASH
    
    output:
        string con las tranformaciones anteriores

    """
    dict = settings.PUNCT_MARK_DICT
    if lowercase: text = text.lower()
    # Elimina signos de puntuación entre caracteres
    text = re.sub(r'(?<![ ])[{}](?![ ])'.format("|".join(map(re.escape, settings.PUNCT_MARKS))),'',text)

    if token_num:
        # Cambiamos los números enteros y decimales con (, o .) por <NUM>, los números presentes en palabras
        # o junto a letras no se cambian
        text = re.sub(r'(?<![,.a-zA-Z0-9-@#$%])[\.|\,]?\d+([\.\,]*\d+)*(?![a-zA-Z0-9-@#$%])',settings.NUM, text)


    # Si not allow_duplicates la aparición de multiples signos como '...' se transforma en '.PERIOD' (uno en lugar de tres)
    if not allow_duplicates:
        mark_list = [*settings.PUNCT_MARKS, '-'] if  token_dash else settings.PUNCT_MARKS
        regex_dup = re.compile(''.join(['|' +re.escape(mark)+'{2,}' for mark in mark_list])[1:])
        text = regex_dup.sub(lambda m: m.group(0)[0], text)

    # Sustituimos los signos de puntuación por los tokens definidos. El punto  se trata a parte para permitir su aparición
    # entre otros carecteres (p.m -> p.m .PERIOD)
    if token_punct:
        punct_marks_or = "|".join(map(re.escape, settings.PUNCT_MARKS[1:]))
        regex_sub = re.compile("(%s)" % (punct_marks_or)) 
        text = regex_sub.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 
        text = re.sub(r'\.(?=[ ])|\.$',dict['.'],text)

    if token_dash:
        # Tokenizamos los guiones
        text = re.sub(r'(?<![a-zA-Z0-9@#$%])-(?![a-zA-Z0-9@#$%])',settings.DASH,text)

    # Añadimos espacios a los signos para poder separarlos con el método split()
    punct_marks_or_token = "|".join(map(re.escape, [*settings.PUNCT_MARKS_TOKENS, settings.DASH]))
    regex_sep = re.compile(punct_marks_or_token) 
    text = regex_sep.sub(lambda m: ' '+m.group(0)+' ', text)
    # Eliminamos espacios duplicados
    text = re.sub(r'[ ]{2,}',' ',text)
    return text




def prepare_file(in_file_path,out_file_name,lowercase=False, allow_duplicates=True,token_punct = True,token_num=True,token_dash = False):
    """
    Aplica la función :func:`prepare_text(...)` a un archivo de texto y genera la salida

    input:
        in_file_path: ruta al archivo de entrada
        out_file_path: ruta al archivo de salida
        resto de parámetros: ver :func:`prepare_text(...)`

    """
    out_file_path = os.path.join(settings.DATA_PREPARED_DIR,out_file_name)
    with open(in_file_path, 'r', encoding='utf-8') as in_file, open(out_file_path, 'w', encoding='utf-8') as out_file:
        in_text = in_file.read()
        out_text = prepare_text(in_text, lowercase, allow_duplicates, token_punct, token_num, token_dash)
        out_file.write(out_text)
    in_file.close()
    out_file.close()
    return out_file_path


def train_dev_test_split(data_path, train_split = 0.7, dev_split = 0.15,shuffle = False):
    """
    Divide un archivo de texto en tres archivos (train, dev y test) necesarios para el modelo de
    Ottokar Tilk y Tanel Alum

    input:
        data_path: ruta al archivo de texto a dividir
        train_split: porcentaje (entre 0 y 1) del archivo original dedicado a la partición train
        dev_split: porcentaje (entre el restante de train_split y 1) del archivo dedicado a dev, 
                   el resto va a test
        shuffle: booleano que indica si se realiza ordenación aleatoria del archivo original

    output:
        rutas a los archivos creados
    """

    assert train_split + dev_split < 1
    with open(data_path, 'r', encoding='utf-8') as data:
        lines = data.readlines()
    data.close()
    prop_split = dev_split / (1- train_split) 
    train, temp = split(lines,train_split,shuffle)
    dev, test = split(temp,prop_split,shuffle)

    name = ''.join(data_path.split(os.sep)[-1].split('.')[:1])
    TRAIN_PREP_PATH = os.path.join(settings.DATA_PREPARED_DIR,name + '.split.train.txt')
    DEV_PREP_PATH = os.path.join(settings.DATA_PREPARED_DIR, name + '.split.dev.txt')
    TEST_PREP_PATH = os.path.join(settings.DATA_PREPARED_DIR, name+'.split.test.txt') 
    with open(TRAIN_PREP_PATH, 'w', encoding='utf-8') as train_out_file,\
         open(DEV_PREP_PATH, 'w', encoding='utf-8') as dev_out_file,\
         open(TEST_PREP_PATH, 'w', encoding='utf-8') as test_out_file:

         train_out_file.write(''.join(train))
         dev_out_file.write(''.join(dev))
         test_out_file.write(''.join(test))
    train_out_file.close()
    dev_out_file.close()
    test_out_file.close()

    return TRAIN_PREP_PATH, DEV_PREP_PATH, TEST_PREP_PATH

def split(X, split=0.7, shuffle = False):
    """
    Función auxiliar que divide una lista en dos en función del porcentaje split
    """
    assert split > 0 and split < 1, 'split must be betweeen 0 and 1'
    size_0 = int(split * len(X))    # Tamaño del primer conjunto            
    indexes = np.arange(len(X))     
    # array de íncides posibles
    if shuffle: np.random.shuffle(indexes)  # reordenación aleatoria de íncides
    # separamos los índices por conjunto
    indexes_0 = indexes[:size_0]                      
    indexes_1 = indexes[size_0:]
    # Definimos los conjutos de entrenamiento y test
    X_0  = [X[i] for i in indexes_0]
    X_1 = [X[i] for i in indexes_1]
    return X_0, X_1



