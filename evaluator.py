import settings
from preprocessor import *

"""
Este módulo contiene las funciones para evaluar los módelos usados en el trabajo, esto es, 
los métodos correspondientes a los Apartados 2, 3 y 5.
"""

#################
# APARTADO 2
#################
def verifyPunctuation(check, test, prepared = False):
    """
    Función principal de Apartado 2. Dados un string check y test, devuelve las operaciones
    necesarias para llegar de test a check (con índice de token referente a check). 
    Las operaciones son:
        * (D,i) (Delete): en test se ha eliminado el token i de check.
        * (I,i) (Insertion): en test se ha insertado un token en la posción i de check.
        * (S,i) (Substitution): en test existe un token distinto con respecto al token i de check.

    """
    # Tokenizamos los textos
    check = tokenizer(check,prepared)
    test = tokenizer(test,prepared)
    punct_marks = settings.PUNCT_MARKS if not prepared else settings.PUNCT_MARKS_TOKENS

    # Hacemos padding
    check, test = padding(check, test)
    l_check = len(check)
    modifications = []   
    
    for i in range(l_check):
        # Deletions:
        # Si test[i] no es un sdp pero check[i] si y las palabras anteriores coinciden salvo mayúsculas
        # añadimos la modificación ('D', i) e insertamos el correspondiente sdp faltante en test.
        if check[i] in punct_marks:
            if test[i] not in punct_marks and test[i-1].upper() == check[i - 1].upper():
                modifications.append(('D',i))
                test.insert(i,check[i])
                
        # Reestablecemos el padding para mantener misma longitud
        check, test = padding(check, test)

        # Insertions:
        # Si test[i] es un sdp pero check[i] no y las palabras anteriores coinciden salvo mayúsculas
        # añadimos la modificación ('I', i) y eliminamos el correspondiente sdp en test.
        if check[i] not in punct_marks:
            if test[i] in punct_marks and test[i-1].upper() == check[i - 1].upper():
                modifications.append(('I',i))
                test.pop(i)
                
    # Reestablecemos el padding para mantener misma longitud
    check, test = padding(check, test)
    
    # Substitutions:
    # Tras haber transformado test para añadir los as sustituciones son aquellos elementos que no coinciden
    for i in range(l_check):
        if check[i] != test[i] and check[i] != '' and test[i] != '':
            modifications.append(('S',i))
    return set(modifications)

#################
# APARTADOS 3 Y 5
#################

def evaluate_example(punctuationFunction, check, test,model = None, add_punct_basic = False, print_info = True, prepared = False):
    """
    Evalúa un método de punctuación haciendo la predicción in situ sobre dos strings. 
    Calcula las siguientes métricas:
        * Precision: proporción de cambios hechos que son correctos
        * Recall: proporción de cambios correctamente hechos sobre los que había que hacer
        * F1: media armónica de los anteriores

    input:
        punctuationFunction: addPunctuationBasic o addPunctuationNgram
        check: string de check
        test: string de test
        model: si addPunctuationNgram, es necesario indicar el modelo de Ngramas
        add_punct_nasic: si addPunctuationNgram, indica si añadir puntuación básica o no

    """
    if model is not None: punctuated = punctuationFunction(model,test,add_punct_basic)
    else: punctuated = punctuationFunction(test)
    punct_marks = settings.PUNCT_MARKS if not prepared else settings.PUNCT_MARKS_TOKENS
    # Diferencia entre el check y el test original (cambios necesarios)
    vct = verifyPunctuation(test,check,prepared)
    # Diferencia entre el test original y el modificado (modificaciones hechas por la función)
    vtmt = verifyPunctuation(test,punctuated,prepared)
    # Diferencia entre el check y el modificado
    vcmt = verifyPunctuation(punctuated,check,prepared)
    es_correcto = 1 if len(vcmt) == 0 else 0
    # número de cambios hechos por la función            
    hechos_ = len(vtmt)
    # número de cambios necesarios
    necesarios_ = len(vct)
    # número de cambios correctos hechos (intersección entre cambios hechos y necesarios)
    correctos_ = len(vtmt & vct) 
    
    # CORRECCIÓN DE ERROR DE SUSTITUCIÓN DE SIGNO
    # Esta definición de correctos_ es incompleta bajo el uso de verifyPunctuation ya que por ejemplo si
    # el modelo añade al final (token 23 por ejemplo) un '.' y debía añadir '?', en ambos casos recibiremos
    # ('I',23) y sin embargo es incorrecto. En resumen, verifyPunctuation no nos devuelve información sobre el 
    # token en si. Corregimos esto con un factor de error.
    error_ = 0
    check_tokens = tokenizer(check,prepared)
    l_t = len(tokenizer(test,prepared))
    l_c = len(check_tokens)
    cambios = list(vtmt)
    diferencias = list(vcmt)
    # Para cada token que aparece en la verificación de la predicción con el test (cambios), comprobamos si 
    # en la verificación de la predicción con el check (diferencias) aparecen operaciones distintas, esto es,
    # se ha insertado un signo incorrecto (I) ya que en la verificación aparece una sustitución de ese signo (S)
    for j in range(len(cambios)):
        token = cambios[j][1]
        operacion = cambios[j][0]
        for k in range(len(diferencias)):
            error_ += 1 if token == diferencias[k][1] and token<l_t and diferencias[k][1]<l_c and operacion == 'I' and diferencias[k][0] == 'S' else 0
    # El token final lo tratamos por separado
    last_punct_token = tokenizer(punctuated,prepared)[-1]
    last_check_token = check_tokens[-1]
    error_ += 1 if last_punct_token in punct_marks and last_check_token in punct_marks and last_punct_token != last_check_token else 0
    # Corregimos correctos_ en base al error
    correctos_ -= error_
    # métricas medias
    precision = (correctos_ / hechos_) if hechos_ != 0 else 0
    recall = (correctos_ / necesarios_) if necesarios_ != 0 else 0
    
    # Si print_info decuelve información sobre el proceso
    if print_info:
        print('TEST LINE: \n ',test)
        print('MODEL PUNCTUATED LINE: \n ',punctuated)
        print('VALIDATION LINE: \n ',check,'\n')
        print('Modificaciones necesarias: ', vct)
        print('Modificaciones hechas por el modelo: ',vtmt)
        print('Diferencias entre modelo y validación: ',vcmt, '\n')
        print('n_hechas (Núm. de modificaciones hechas por el modelo): ',hechos_)
        print('n_correctas (Núm. de modificaciones correctas, i.e., \n interseccion(hechas,necesarias) - error de sustitucion de signo): ',correctos_)
        print('n_necesarias (Núm. de modificaciones necesarias): ',necesarios_, '\n')
        print('precision (n_correctas/n_hechas): ',precision)
        print('recall (n_correctas/n_necesarias): ',recall,'\n')
    
    return es_correcto, precision, recall, hechos_, correctos_, necesarios_


def evaluate(punctuationFunction, check_file_path , test_file_path , model = None, add_punct_basic = False, prepared = False):

    """
    Función evaluadiora principal. Evalúa un método de punctuación haciendo la predicción in situ sobre dos archivos.
    Calcula las métricas Precisión, Recall y F1 tanto globales (sobre todas las intancias) como medias (medias de las 
    medidas de cada instancia evaluada).

    input:
        punctuationFunction: addPunctuationBasic o addPunctuationNgram
        check_file_path: ruta a archivo de check
        test_file_path: ruta a archivo de test
        model: si addPunctuationNgram, es necesario indicar el modelo de Ngramas
        add_punct_nasic: si addPunctuationNgram, indica si añadir puntuación básica o no
    """
    num_correctos = 0   # número de instancias correctamente puntuadas
    hechos = 0          # número de cambios globales hechos por la función  
    correctos = 0       # número de modificaciones correctas globales
    necesarios = 0      # número de modificaciones necesarias globales
    
    
    with open(test_file_path, 'r', encoding='utf-8-sig') as test_file,\
         open(check_file_path, 'r', encoding='utf-8-sig') as check_file:
        
        test_lines = test_file.readlines() 
        check_lines = check_file.readlines() 
           
        # precisión y recall por instancia para calcular las medias
        precision_, recall_ = [], []
        # número de instancias
        N = len(test_lines)
        #testeo = True
        for i in range(N):
            check = check_lines[i].rstrip(' \n')
            test = test_lines[i].rstrip(' \n')
            es_correcto, prec, rec, hechos_, correctos_, necesarios_ =\
                evaluate_example(punctuationFunction,check,test,model=model,\
                                 add_punct_basic = add_punct_basic,print_info=False, prepared = prepared)
            
            # métricas por línea
            num_correctos += es_correcto
            precision_.append(prec)
            recall_.append(rec)
            hechos += hechos_
            correctos += correctos_
            necesarios += necesarios_
            
        # Métricas medias
        precision_media = sum(precision_)/len(precision_)
        recall_media = sum(recall_)/len(recall_)
        F1_media = 2 * (precision_media * recall_media) / (precision_media + recall_media)
        rendimiento = num_correctos/N
    
    # Cerramos los streams de datos
    test_file.close()
    check_file.close()
            
    # Métricas globales
    precision = correctos / hechos
    recall = correctos / necesarios
    F1 = 2 * (precision * recall) / (precision + recall)
   
    w = 45
    print('='*w)
    print('MÉTRICAS')
    print('='*w)
    print ('precision global: ', precision)
    print ('recall global: ', recall)
    print ('F1 global: ', F1)
    print('='*w)
    print ('precision media: ', precision_media)
    print ('recall medio: ', recall_media)
    print ('F1 medio: ', F1_media)
    print('='*w)
    print ('rendimiento: ', rendimiento)
    print('='*w)
    print('número de instancias en el corpus: ',N)
    print('='*w)
    
    result_dict = {'precision_global':precision, 'recall_global':recall, 'F1_global':F1,'precision_mean':precision_media, 'recall_mean':recall_media, 'F1_mean':F1_media, 'score':rendimiento }
    return result_dict
    
def evaluate_example_from_corpus(punctuationFunction, check_file_path, test_file_path,corpus_line = 0, model = None,add_punct_basic = False, prepared = False):
    """
    Esta función sirve simplemente para explorar el corpus comparando el resultado de la función de puntuación.
        
    input:    
        corpus_line: número de línea que se desea explorar.
        resto de parámetros: ver función :func:`evaluate`.
    """
    with open(test_file_path, 'r', encoding='utf-8-sig') as test_file, open(check_file_path, 'r', encoding='utf-8-sig') as check_file:
        check = check_file.readlines()[corpus_line].rstrip(' \n')
        test = test_file.readlines()[corpus_line].rstrip(' \n')
        evaluate_example(punctuationFunction,check,test,model = model,add_punct_basic = add_punct_basic,print_info=True, prepared=prepared)
    test_file.close()
    check_file.close()


"""
Las siguientes funciones son análogas a las anteriores pero en lugar de hacer la predicción in situ
tienen de entrada las rutas al archivo de check y test YA PUNTUADO por algún modelo.
"""
def evaluate_example_punctuated(check, test, punctuated, print_info = True, prepared = False, readeable = False):
    """
    Análoga a :func:`evaluate_example` pero 'test' es un string ya puntuado por algún modelo.
    """
    punct_marks = settings.PUNCT_MARKS if not prepared else settings.PUNCT_MARKS_TOKENS
    # Diferencia entre el check y el test original (cambios necesarios)
    vct = verifyPunctuation(test,check,prepared)
    # Diferencia entre el test original y el modificado (modificaciones hechas por la función)
    vtmt = verifyPunctuation(test,punctuated,prepared)
    # Diferencia entre el check y el modificado (modificaciones hechas por la función)
    vcmt = verifyPunctuation(punctuated,check,prepared)
    es_correcto = 1 if len(vcmt) == 0 else 0
    # número de cambios hechos por la función            
    hechos_ = len(vtmt)
    # número de cambios necesarios
    necesarios_ = len(vct)
    # número de cambios correctos hechos (intersección entre cambios hechos y necesarios)
    correctos_ = len(vtmt & vct) 
    
    # CORRECCIÓN DE ERROR DE SUSTITUCIÓN DE SIGNO
    # Esta definición de correctos_ es incompleta bajo el uso de verifyPunctuation ya que por ejemplo si
    # el modelo añade al final (token 23 por ejemplo) un '.' y debía añadir '?', en ambos casos recibiremos
    # ('I',23) y sin embargo es incorrecto. En resumen, verifyPunctuation no nos devuelve información sobre el 
    # token en si. Corregimos esto con un factor de error.
    error_ = 0
    check_tokens = tokenizer(check,prepared)
    l_t = len(tokenizer(test,prepared))
    l_c = len(check_tokens)
    cambios = list(vtmt)
    diferencias = list(vcmt)
    # Para cada token que aparece en la verificación de la predicción con el test (cambios), comprobamos si 
    # en la verificación de la predicción con el check (diferencias) aparecen operaciones distintas, esto es,
    # se ha insertado un signo incorrecto (I) ya que en la verificación aparece una sustitución de ese signo (S)
    for j in range(len(cambios)):
        token = cambios[j][1]
        operacion = cambios[j][0]
        for k in range(len(diferencias)):
            error_ += 1 if token == diferencias[k][1] and token<l_t and diferencias[k][1]<l_c and operacion == 'I' and diferencias[k][0] == 'S' else 0
    # El token final lo tratamos por separado
    last_punct_token = tokenizer(punctuated,prepared)[-1]
    last_check_token = check_tokens[-1]
    error_ += 1 if last_punct_token in punct_marks and last_check_token in punct_marks and last_punct_token != last_check_token else 0
    # Corregimos correctos_ en base al error
    correctos_ -= error_
    # métricas medias
    precision = (correctos_ / hechos_) if hechos_ != 0 else 0
    recall = (correctos_ / necesarios_) if necesarios_ != 0 else 0
    
    def to_readeable(text):
        dict = settings.PUNCT_MARK_DICT
        rev_dict = inv_map = {v: k for k, v in dict.items()}
        return "".join(rev_dict.get(w,' '+w) for w in text.split())[1:]       
         
    if readeable:
        punctuated = to_readeable(punctuated)
        check = to_readeable(check)

    # Si print_info decuelve información sobre el proceso
    if print_info:
        print('TEST LINE: \n ',test)
        print('MODEL PUNCTUATED LINE: \n ',punctuated)
        print('VALIDATION LINE: \n ',check,'\n')
        print('Modificaciones necesarias: ', vct)
        print('Modificaciones hechas por el modelo: ',vtmt)
        print('Diferencias entre modelo y validación: ',vcmt, '\n')
        print('n_hechas (Núm. de modificaciones hechas por el modelo): ',hechos_)
        print('n_correctas (Núm. de modificaciones correctas: \n interseccion(hechas,necesarias) - error de sustitucion de signo): ',correctos_)
        print('n_necesarias (Núm. de modificaciones necesarias): ',necesarios_, '\n')
        print('precision (n_correctas/n_hechas): ',precision)
        print('recall (n_correctas/n_necesarias): ',recall,'\n')
    
    return es_correcto, precision, recall, hechos_, correctos_, necesarios_


def evaluate_punctuated(check_file_path , test_file_path, punctuated_file_path, prepared = False):
    """
    Análoga a :func:`evaluate` pero 'test_file_path' es es la ruta a un archivo 
    ya puntuado por algún modelo.
    """

    # número de instancias correctamente puntuadas
    num_correctos = 0
    # número de cambios globales hechos por la función  
    hechos = 0
    # número de modificaciones correctas globales
    correctos = 0
    # número de modificaciones necesarias globales
    necesarios = 0
    
    
    with open(test_file_path, 'r', encoding='utf-8-sig') as test_file,\
         open(punctuated_file_path, 'r', encoding='utf-8-sig') as punc_file,\
         open(check_file_path, 'r', encoding='utf-8-sig') as check_file:
        
        test_lines = test_file.readlines() 
        check_lines = check_file.readlines()
        punc_lines = punc_file.readlines()
        
       
        # precisión y recall por instancia para calcular las medias
        precision_, recall_ = [], []
        # número de instancias
        N = len(test_lines)
        #testeo = True
        for i in range(N):
            check = check_lines[i].rstrip(' \n')
            test = test_lines[i].rstrip(' \n')
            punc = punc_lines[i].rstrip(' \n')
            es_correcto, prec, rec, hechos_, correctos_, necesarios_ = evaluate_example_punctuated(check,test,punc,print_info=False, prepared=prepared)
            
            # métricas por línea
            num_correctos += es_correcto
            precision_.append(prec)
            recall_.append(rec)
            hechos += hechos_
            correctos += correctos_
            necesarios += necesarios_
            
        # Métricas medias
        precision_media = sum(precision_)/len(precision_)
        recall_media = sum(recall_)/len(recall_)
        F1_media = 2 * (precision_media * recall_media) / (precision_media + recall_media)
        rendimiento = num_correctos/N
    
    # Cerramos los streams de datos
    test_file.close()
    check_file.close()
            
    # Métricas globales
    precision = correctos / hechos
    recall = correctos / necesarios
    F1 = 2 * (precision * recall) / (precision + recall)
   
    w = 45
    print('='*w)
    print('MÉTRICAS')
    print('='*w)
    print ('precision global: ', precision)
    print ('recall global: ', recall)
    print ('F1 global: ', F1)
    print('='*w)
    print ('precision media: ', precision_media)
    print ('recall medio: ', recall_media)
    print ('F1 medio: ', F1_media)
    print('='*w)
    print ('rendimiento: ', rendimiento)
    print('='*w)
    print('número de instancias en el corpus: ',N)
    print('='*w)

    result_dict = {'precision_global':precision, 'recall_global':recall, 'F1_global':F1,'precision_mean':precision_media, 'recall_mean':recall_media, 'F1_mean':F1_media, 'score':rendimiento }
    return result_dict

def evaluate_example_from_corpus_punctuated(check_file_path, test_file_path, punctuated_file_path,corpus_line = 0, prepared = False, readeable = False):
    """
    Análoga a :func:`evaluate_from_corpus` pero 'test_file_path' es es la ruta a un archivo 
    ya puntuado por algún modelo.
    """
    with open(test_file_path, 'r', encoding='utf-8-sig') as test_file, open(check_file_path, 'r', encoding='utf-8-sig') as check_file,\
         open(punctuated_file_path, 'r', encoding='utf-8-sig') as punctuated_file:
        check = check_file.readlines()[corpus_line].rstrip(' \n')
        test = test_file.readlines()[corpus_line].rstrip(' \n')
        punctuated = punctuated_file.readlines()[corpus_line].rstrip(' \n')
        evaluate_example_punctuated(check,test, punctuated,print_info=True, prepared=prepared, readeable=readeable)
    test_file.close()
    check_file.close()