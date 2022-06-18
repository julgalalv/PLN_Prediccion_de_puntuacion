import settings
from preprocessor import change_initial

# Función del apartado 1. Pone en maúscula la primera letra y añade un punto '.' al final si no lo tiene
def addPunctuationBasic(string):
    initial_upper = change_initial(string,uppercase=True)
    last_char = initial_upper[-1]
    add_dot = '.' if last_char not in settings.PUNCT_MARKS else ''
    return initial_upper + add_dot