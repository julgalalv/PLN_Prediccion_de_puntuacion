from preprocessor import *

def addPunctuationNgram(model,example,add_basic_punct = False):
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
    