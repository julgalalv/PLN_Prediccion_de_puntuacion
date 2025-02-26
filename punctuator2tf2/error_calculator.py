# coding: utf-8

"""
Computes and prints the overall classification error and precision, recall, F-score over punctuations.
"""

from numpy import nan
import data
import codecs
import sys

"""
Archivo adaptado para el Trabajo de Predicción de Puntuación de la asignatura Procecsamiento del Lenguaje Natural del Máster MULCIA
    Repositorio original: https://github.com/ottokart/punctuator2
    Reimplementación de los autores en tensorflor : https://github.com/cadia-lvl/punctuation-prediction/tree/master/punctuator2tf2
"""

# sustituciones de signos de puntuación (opcional)
MAPPING = {}
#MAPPING = {"!EXCLAMATIONMARK": ".PERIOD", ":COLON": ".COMMA", ";SEMICOLON": ".PERIOD", "-DASH": ",COMMA"}


def compute_error(target_paths, predicted_paths):
    """
    Calcula las métricas precision, recall y F1 dadas las rutas de archivos de check (target) y puntuados por el modelo
    y las imprime por consola
    """

    # lee el vocabulario de puntuaciones desde el archivo
    punctuation_vocabulary = data.read_vocabulary(data.PUNCT_VOCAB_FILE)

    counter = 0             # contador de tokens procesados
    total_correct = 0       # contador de puntuaciones correctas

    correct = 0.            # contador de correcciones correctas por línea
    substitutions = 0.      # contador de error por sustituciones hechas
    deletions = 0.          # contador de error por eliminaciones hechas
    insertions = 0.         # contador de error por inserciones hechas

    true_positives = {}     # diccionario con TP (verdaderos positivos)
    false_positives = {}    # diccionario con FP (falsos positivos)
    false_negatives = {}    # diccionario con FN (falsos negativos)

    for target_path, predicted_path in zip(target_paths, predicted_paths):

        target_punctuation = " "        # signo de puntuación procesado en la iteración en check 
        predicted_punctuation = " "     # signo de puntuación procesado en la iteración en test

        # Índices de posicion en secuencias check  y test
        t_i = 0     # check (target)
        p_i = 0     # test (punctuation)

        with codecs.open(target_path, 'r', 'utf-8') as target, codecs.open(predicted_path, 'r', 'utf-8') as predicted:

            # se leen ambos archivos completamente y se separan por espacios
            target_stream = target.read().split()
            predicted_stream = predicted.read().split()
            
            while True:
                # si el elemento t_i es signo de puntuación, se mapea si procede y si son consecutivos se queda con el primero
                # estableciéndolo como signo actual
                if data.PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i]) in punctuation_vocabulary:
                    while data.PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i]) in punctuation_vocabulary: # skip multiple consecutive punctuations
                        target_punctuation = data.PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i])
                        target_punctuation = MAPPING.get(target_punctuation, target_punctuation)
                        t_i += 1
                else:
                    target_punctuation = " "    # en caso contrario se mantiene el placeholder

                # Si el elemeto p_i es de puntuacion se establece como el signo de puntuación actual 
                if predicted_stream[p_i] in punctuation_vocabulary:
                    predicted_punctuation = MAPPING.get(predicted_stream[p_i], predicted_stream[p_i])
                    p_i += 1
                else:
                    predicted_punctuation = " "

                # si los signos de check y test son iguales, es correcto
                is_correct = target_punctuation == predicted_punctuation

                counter += 1 
                total_correct += is_correct

                # se considera una eliminación si en test hay un espacio y en check no
                if predicted_punctuation == " " and target_punctuation != " ":
                    deletions += 1
                # se considera una inserción si en test no hay un espacio y en check si
                elif predicted_punctuation != " " and target_punctuation == " ":
                    insertions += 1
                # si coinciden y no son espacios, es correcto
                elif predicted_punctuation != " " and target_punctuation != " " and predicted_punctuation == target_punctuation:
                    correct += 1
                # si no coinciden y no son espacios, es una sustitución
                elif predicted_punctuation != " " and target_punctuation != " " and predicted_punctuation != target_punctuation:
                    substitutions += 1

                # Para cada signo de puntuacion se establecen TP, FP y FN

                # TP si el signo predicho corresponde con el del target
                true_positives[target_punctuation] = true_positives.get(target_punctuation, 0.) + float(is_correct)
                # FP si se ha añadido una predicción que no era la del target
                false_positives[predicted_punctuation] = false_positives.get(predicted_punctuation, 0.) + float(not is_correct)
                # FN si no se ha añadido una predicción necesaria
                false_negatives[target_punctuation] = false_negatives.get(target_punctuation, 0.) + float(not is_correct)

                # se comprueba que el resto de palabras coinciden. Por ello la comparación se hace en minúscula (ya que el modelo)
                # no capitaliza por defecto
                assert target_stream[t_i].lower() == predicted_stream[p_i].lower() or predicted_stream[p_i] == "<unk>", \
                        ("File: %s \n" + \
                        "Error: %s (%s) != %s (%s) \n" + \
                        "Target context: %s \n" + \
                        "Predicted context: %s") % \
                        (target_path,
                        target_stream[t_i], t_i, predicted_stream[p_i], p_i,
                        " ".join(target_stream[t_i-2:t_i+2]),
                        " ".join(predicted_stream[p_i-2:p_i+2]))
                

                
                t_i += 1
                p_i += 1

                if t_i >= len(target_stream)-1 and p_i >= len(predicted_stream)-1:
                    break
    
    # TP, FP, FN globales
    overall_tp = 0.0
    overall_fp = 0.0
    overall_fn = 0.0

    print("-"*46)
    print("{:<16} {:<9} {:<9} {:<9}".format('PUNCTUATION','PRECISION','RECALL','F-SCORE'))
    for p in punctuation_vocabulary:

        if p == data.SPACE:
            continue

        overall_tp += true_positives.get(p,0.)
        overall_fp += false_positives.get(p,0.)
        overall_fn += false_negatives.get(p,0.)

        # Se calculan la precision, recall y f_score y se muestran por pantalla
        punctuation = p
        precision = (true_positives.get(p,0.) / (true_positives.get(p,0.) + false_positives[p])) if p in false_positives else nan
        recall = (true_positives.get(p,0.) / (true_positives.get(p,0.) + false_negatives[p])) if p in false_negatives else nan
        f_score = (2. * precision * recall / (precision + recall)) if (precision + recall) > 0 else nan        
        print("{:<16} {:<9.3f} {:<9.3f} {:<9.3f}".format(punctuation, round(precision,3)*100, round(recall,3)*100, round(f_score,3)*100))
    print("-"*46)
    pre = overall_tp/(overall_tp+overall_fp) if overall_fp else nan
    rec = overall_tp/(overall_tp+overall_fn) if overall_fn else nan
    f1 = (2.*pre*rec)/(pre+rec) if (pre + rec) else nan
    print("{:<16} {:<9.3f} {:<9.3f} {:<9.3f}".format("Overall", round(pre,3)*100, round(rec,3)*100, round(f1,3)*100))
    print("Err: %s%%" % round((100.0 - float(total_correct) / float(counter-1) * 100.0), 2))
    print("SER: %s%%" % round((substitutions + deletions + insertions) / (correct + substitutions + deletions) * 100, 1))


if __name__ == "__main__":

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        sys.exit("Ground truth file path argument missing")

    if len(sys.argv) > 2:
        predicted_path = sys.argv[2]
    else:
        sys.exit("Model predictions file path argument missing")

    compute_error([target_path], [predicted_path])    
        
