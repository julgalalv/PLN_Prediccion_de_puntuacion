from preprocessor import *

"""
Funciones del Apartado 4 correspondientes a la puntuación basada en 4-gramas
"""

def addPunctuationNgram(model,example,add_basic_punct = False):
    """
    Función principal del Apartado 4. Usa un modelo preentrenado de clase :class:`ModelNgram`
    para puntuar un string

    input:
        model: modelo preentrenado de clase :class:`ModelNgram` 
        example: string a puntuar
        add_basic_punct: si True se realiza puntuación básica poniendo la primera letra en mayúsculas y 
                         un punto al final si no hay otro signo.
    output:
        string puntuado.


    """
    # Comprobamos que el modelo ha sido entrenado
    model.check_entrenado()
    # Trabajamos con los tokens
    tokens = tokenizer(example)
    num_tokens = len(tokens)
    # Generamos los (N-1)-gramas del texto
    N = model.N -1
    grams = ngrams(example,N)
    added_tokens = 0
    for i in range(len(grams)):
        # Calculamos el 4-grama predicho
        operation = model.predice(grams[i]) if N > 1 else model.predice(grams[i][0])
        target_index = i+N+added_tokens
        # Transformamos los tokens del texto
        if operation == model.mayus and target_index < num_tokens:
            tokens[target_index] = change_initial(tokens[target_index], uppercase = True)
        if operation == model.minus and target_index < num_tokens:
            tokens[target_index] = change_initial(tokens[target_index], uppercase = False)
        if operation in model.punct_marks:
            added_tokens += 1
            num_tokens += 1
            tokens.insert(target_index, operation)
            if operation in model.mayus_marks and target_index < num_tokens -1:
                tokens[target_index+1] = change_initial(tokens[target_index+1], uppercase = True)

    # Añadimos los espacios excepto para los signos de puntuación
    result = [' ' + x if x not in model.punct_marks else x for x in tokens]
    # Reconstruimos el texto predicho
    result = ''.join(result)[1:]
    
    # Puesto que el modelo de 4 gramas dificilmente va a poner la primera letra en mayúscula y es probable que
    # deje el final de la oración sin puntuar, añadimos la funcionalidad de addPunctuationBasic como
    # parámetro de la función.
    if add_basic_punct:
        dot = '' if result[-1] in model.punct_marks else '.'
        result = change_initial(result + dot,uppercase=True)
    return result 
    
def addPunctuationNgram_file(model,in_file_path, out_file_name='', line_by_line = True, add_basic_punct = False):
    """
    Aplica :func:`addPunctuationNgram` a un archivo de texto y genera la salida en otro archivo

    input:
        model: modelo preentrenado de clase :class:`ModelNgram` 
        in_file_path: ruta del archivo de entrada
        out_file_path: ruta del archivo de salida
        line_by_line: si True puntúa línea a línea o todo el texto 
        add_basic_punct: si True se realiza puntuación básica poniendo la primera letra en mayúsculas y 
                         un punto al final si no hay otro signo.
    
    output:
        None
    """
    out_file_path = os.path.join(settings.PREDICTED_DIR, out_file_name +'.ngram_punctuator_predicted.txt')
    with open(in_file_path, 'r', encoding='utf-8') as in_file,\
         open(out_file_path, 'w',encoding='utf-8') as out_file:
         in_lines = in_file.readlines() if line_by_line else in_file.read()
         if isinstance(in_lines,list):
            for line in in_lines:
                line = line.rstrip()
                out_file.write(addPunctuationNgram(model,line, add_basic_punct))
         else:
            out_file.write(addPunctuationNgram(model,in_lines,add_basic_punct))
    in_file.close()
    out_file.close()