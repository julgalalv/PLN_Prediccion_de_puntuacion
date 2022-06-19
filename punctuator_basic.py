import settings
import os
from preprocessor import change_initial


"""
Funciones del Apartado 1 correspondientes a la puntuación básica
"""
def addPunctuationBasic(string):
    """
    Función principal del Apartado 1. Pone en mayúscula la primera letra 
    del string  y añade un punto '.' al final si no lo tiene.

    Devuelve el string puntuado.
    """
    initial_upper = change_initial(string,uppercase=True)
    last_char = initial_upper[-1]
    add_dot = '.' if last_char not in settings.PUNCT_MARKS else ''
    return initial_upper + add_dot

def addPunctuationBasic_file(in_file_path, out_file_name='', line_by_line = True):
    """
    Aplica :func:`addPunctuationBasic` a un archivo de texto y genera la salida en otro archivo

    input:
        in_file_path: ruta del archivo de entrada
        out_file_path: ruta del archivo de salida
        line_by_line: si True puntúa línea a línea o todo el texto 
    
    output:
        None
    """
    out_file_path = os.path.join(settings.PREDICTED_DIR, out_file_name +'.basic_punctuator_predicted.txt')
    with open(in_file_path, 'r', encoding='utf-8') as in_file,\
         open(out_file_path, 'w',encoding='utf-8') as out_file:
         in_lines = in_file.readlines() if  line_by_line else in_file.read()
         if line_by_line:
            for line in in_lines:
                out_file.write(addPunctuationBasic(line.rstrip()))
         else:
            line = in_lines.replace(' \n', ' ') if flatten else in_lines
            out_file.write(addPunctuationBasic(line))
    in_file.close()
    out_file.close()