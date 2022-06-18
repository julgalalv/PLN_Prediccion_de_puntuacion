import re
import os
import settings
import numpy as np

# Función que pone el primer caracter de un string en mayúscula o minúscula según el parámetro booleano 'uppercase'
def change_initial(string,uppercase):
    if not string:
        return
    init = string[0]
    new_initial = init.upper() if uppercase else init.lower()
    temp = list(string)
    temp[0] = new_initial
    return ''.join(temp)

# Función que tokeniza un texto
def tokenizer(text,prepare = False):
    if not prepare:
        # Generamos la expresión regular que añade espacios a los signos para poder separarlos con el método split()
        marks = ''.join(settings.PUNCT_MARKS)
        regex1, regex2 = r' ?([{}]+) ?'.format(marks), r' \1 '
        text = re.sub(regex1, regex2, text)
    return text.split()
 
def ngrams(text,N):
    text = tokenizer(text)
    return [tuple(text[i:i+N]) for i in range(len(text)-N+1)]


# Dadas dos listas, añade '' a la menor hasta que ambas tengan el mismo tamaño.
def padding(list1, list2):
    len1, len2 = len(list1), len(list2)
    max_len = max(len1, len2)
    list1 = [*list1, *([''] * (max_len - len1))]
    list2 = [*list2, *([''] * (max_len - len2))]
    return list1, list2

def prepare_text(text, lowercase=False, allow_duplicates=True):
    dict = settings.PUNCT_MARK_DICT
    if lowercase: text = text.lower()
    # Cambiamos los números enteros y decimales con (, o .) por <NUM>, los números presentes en palabras
    # O junto a letras no se cambian
    text = re.sub(r'(?<![a-zA-Z0-9-@#$%])\d*[\.\,]*\d+(?![a-zA-Z0-9-@#$%])',settings.NUM, text)
    punct_marks_or = "|".join(map(re.escape, settings.PUNCT_MARKS[1:]))

    # Si not allow_duplicates la aparición de multiples signos como '...' se transforma en '.PERIOD' (uno en lugar de tres)
    if not allow_duplicates:
        regex_dup = re.compile(''.join(['|' +re.escape(mark)+'{2,}' for mark in [*settings.PUNCT_MARKS, '-']])[1:])
        text = regex_dup.sub(lambda m: m.group(0)[0], text)

    # Sustituimos los signos de puntuación por los tokens definidos. El punto se trata a parte para permitir su aparición
    # entre otros carecteres (p.m -> p.m .PERIOD)
    regex_sub1 = re.compile("(%s)" % (punct_marks_or)) 
    text = regex_sub1.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 
    text = re.sub(r'\.(?=[ ])|\.$',dict['.'],text)

    # Tokenizamos los guiones
    text = re.sub(r'(?<![a-zA-Z0-9@#$%])-(?![a-zA-Z0-9@#$%])',settings.DASH,text)

    # Añadimos espacios a los signos para poder separarlos con el método split()
    punct_marks_or_token = "|".join(map(re.escape, [*settings.PUNCT_MARKS_TOKENS, settings.DASH]))
    regex_sep = re.compile(punct_marks_or_token) 
    text = regex_sep.sub(lambda m: ' '+m.group(0)+' ', text)
    # Eliminamos espacios duplicados
    text = re.sub(r'[ ]{2,}',' ',text)
    return text

def prepare_file(in_file_path,out_file_name,lowercase=False, allow_duplicates=True):
    out_file_path = os.path.join(settings.DATA_PREPROCESSED_DIR,out_file_name)
    with open(in_file_path, 'r', encoding='utf-8') as in_file, open(out_file_path, 'w', encoding='utf-8') as out_file:
        in_text = in_file.read()
        out_text = prepare_text(in_text,lowercase=lowercase,allow_duplicates=allow_duplicates)
        out_file.write(out_text)
    in_file.close()
    out_file.close()
    return out_file_path

#train_path = os.path.join(settings.DATA_RAW_DIR,'PunctuationTask.train.en')
#out_path =  os.path.join(settings.DATA_PREPROCESSED_DIR,'data.train.txt')
#prepare_file(train_path,out_path)

settings.initialize()

def train_dev_test_split(data_path, train_split = 0.7, dev_split = 0.15):
    assert train_split + dev_split < 1
    with open(data_path, 'r', encoding='utf-8') as data:
        lines = data.readlines()
    data.close()
    prop_split = dev_split / (1- train_split) 
    train, temp = split(lines,train_split)
    dev, test = split(temp,prop_split)

    TRAIN_PREP_PATH = os.path.join(settings.DATA_PREPARED_DIR,'train_split.train.txt')
    DEV_PREP_PATH = os.path.join(settings.DATA_PREPARED_DIR,'train_split.dev.txt')
    TEST_PREP_PATH = os.path.join(settings.DATA_PREPARED_DIR,'train_split.test.txt') 
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

TTTT = './data/preprocessed/train.prepared.txt'
def split(X, split=0.7, shuffle = False):
    assert split > 0 and split < 1, 'split must be betweeen 0 and 1'
    size_0 = int(split * len(X))                         # Tamaño del conjunto de test             
    indexes = np.arange(len(X))     
    # array de íncides posibles
    if shuffle: np.random.shuffle(indexes)                              # reordenación aleatoria de íncides
    # separamos los índices por conjunto
    indexes_0 = indexes[:size_0]                      
    indexes_1 = indexes[size_0:]
    # Definimos los conjutos de entrenamiento y test
    X_0  = [X[i] for i in indexes_0]
    X_1 = [X[i] for i in indexes_1]
    return X_0, X_1

A0, A1 = split(TTTT,0.7)
print(len(TTTT))
print(len(A0))
print(len(A1))
TRAIN_TRAIN_PREP_PATH, TRAIN_DEV_PREP_PATH, TRAIN_TEST_PREP_PATH = train_dev_test_split(TTTT, train_split = 0.7, dev_split = 0.15)



