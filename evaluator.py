import settings
from preprocessor import *

#Función veryfyPunctuation del apartado 2.
def verifyPunctuation(check, test):
    # Tokenizamos los textos
    check = tokenizer(check)
    test = tokenizer(test)
    # Hacemos padding
    check, test = padding(check, test)
    l_check = len(check)
    modifications = []   
    
    for i in range(l_check):
        # Deletions:
        # Si test[i] no es un sdp pero check[i] si y las palabras anteriores coinciden salvo mayúsculas
        # añadimos la modificación ('D', i) e insertamos el correspondiente sdp faltante en test.
        if check[i] in settings.PUNCT_MARKS:
            if test[i] not in settings.PUNCT_MARKS and test[i-1].upper() == check[i - 1].upper():
                modifications.append(('D',i))
                test.insert(i,check[i])
                
        # Reestablecemos el padding para mantener misma longitud
        check, test = padding(check, test)

        # Insertions:
        # Si test[i] es un sdp pero check[i] no y las palabras anteriores coinciden salvo mayúsculas
        # añadimos la modificación ('I', i) y eliminamos el correspondiente sdp en test.
        if check[i] not in settings.PUNCT_MARKS:
            if test[i] in settings.PUNCT_MARKS and test[i-1].upper() == check[i - 1].upper():
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

def evaluate_example(punctuationFunction, check, test,model = None, add_punct_basic = False, print_info = True):
    if model is not None: punctuated = punctuationFunction(model,test,add_punct_basic)
    else: punctuated = punctuationFunction(test)
    # Diferencia entre el check y el test original (cambios necesarios)
    vct = verifyPunctuation(test,check)
    # Diferencia entre el test original y el modificado (modificaciones hechas por la función)
    vtmt = verifyPunctuation(test,punctuated)
    # Diferencia entre el check y el modificado (modificaciones hechas por la función)
    vcmt = verifyPunctuation(punctuated,check)
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
    l_t = len(tokenizer(test))
    l_c = len(tokenizer(check))
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
    error_ += 1 if punctuated[-1] in settings.PUNCT_MARKS and check[-1] in settings.PUNCT_MARKS and punctuated[-1] != check[-1] else 0
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
        print('n_correctas (Núm. de modificaciones correctas: \n interseccion(hechas,necesarias) - error de sustitucion de signo): ',correctos_)
        print('n_necesarias (Núm. de modificaciones necesarias): ',necesarios_, '\n')
        print('precision (n_correctas/n_hechas): ',precision)
        print('recall (n_correctas/n_necesarias): ',recall,'\n')
    
    return es_correcto, precision, recall, hechos_, correctos_, necesarios_

    # Esta función calcula las métricas de una función de puntuación. los atributos son:
#   punctuationFunction: función a evaluar.
#   check_file_path y test_file_path: rutas a los archivos de check y test.
#   model: modelo de predicción usado (4gramas más adelante)
#   add_punct_basic se usa para la función addPunctuation4gram que definiremos más adelante
def evaluate(punctuationFunction, check_file_path , test_file_path , model = None, add_punct_basic = False, all_file = False):
    # número de instancias correctamente puntuadas
    num_correctos = 0
    # número de cambios globales hechos por la función  
    hechos = 0
    # número de modificaciones correctas globales
    correctos = 0
    # número de modificaciones necesarias globales
    necesarios = 0
    
    
    with open(test_file_path, 'r', encoding='utf-8-sig') as test_file,\
         open(check_file_path, 'r', encoding='utf-8-sig') as check_file:
        
        test_lines = test_file.readlines() if not all_file else test_file.read()
        check_lines = check_file.readlines() if not all_file else check_file.read()
        
        # Compara los textos check y text completos. 
        # CUIDADO: Tarda más que línea a línea ya que VerifyPunctuation es costosa
        if all_file:
            es_correcto, prec, rec, hechos_, correctos_, necesarios_ =\
                evaluate_example(punctuationFunction,check_lines,test_lines,model=model,\
                                 add_punct_basic = add_punct_basic,print_info=False)
            # métricas medias
            hechos = hechos_
            correctos = correctos_
            necesarios = necesarios_
            
        # Compara los textos check y text línea a línea
        else:    
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
                                     add_punct_basic = add_punct_basic,print_info=False)
                
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
    if not all_file:
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
    
def evaluate_example_from_corpus(punctuationFunction, check_file_path, test_file_path,corpus_line = 0, model = None,add_punct_basic = False):
    with open(test_file_path, 'r', encoding='utf-8-sig') as test_file, open(check_file_path, 'r', encoding='utf-8-sig') as check_file:
        check = check_file.readlines()[corpus_line].rstrip(' \n')
        test = test_file.readlines()[corpus_line].rstrip(' \n')
        evaluate_example(punctuationFunction,check,test,model = model,add_punct_basic = add_punct_basic,print_info=True)
    test_file.close()
    check_file.close()