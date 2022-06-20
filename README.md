# PLN_Prediccion_de_puntuacion

Proyecto correspondiente a la Tarea 2 evaluable de la asignatura de Procesamiento del Lenguaje Natural impartida en el Máster de Lógica, Computación e Inteligencia Artificial de la Universidad de Sevilla

# Ejercicio práctico: Predicción de puntuación

## Introducción

Este trabajo aborda la tarea de ''Predicción de puntuación'', donde el objetivo va a ser definir una serie de modelos que puedan introducir puntuación en una entrada que carece de puntuación.

En concreto, los signos de puntuación que se van a considerar son:
 * Punto: .
 * Coma: ,
 * Punto y coma: :
 * Dos puntos: :
 * Signo de exclamación: !
 * Signo de interrogación: ?

A lo largo del trabajo se vana  definir dos modelos, uno de puntuación básica y otro basado en 4-gramas. Además se va a definir una serie de funciones para evaluar los modelos basadas en una función implementada (verifyPunctuation), basándose en la distancia de Levenshtein.

## Presentación del trabajo

El trabajo está pensado para ser visto en tres partes y en el siguiente orden:
 1. Apartados 1 a 5: Notebook TRABAJO_PLN_APARTADOS_1_5_JULIAN_GALINDO_ALVAREZ_20091281E.ipynb
 1. Apartado 6: Notebook TRABAJO_PLN_APARTADO_6_JULIAN_GALINDO_ALVAREZ_20091281E.ipynb
 1. Apartado 7: Documento TRABAJO_PLN_APARTADO_7_JULIAN_GALINDO_ALVAREZ_20091281E.pdf

Los notebooks se apoyan en módulos python definidos en el repositorio, que voy importando conforme son necesarios. Por ello recomiendo recorrer los notebooks junto al código del repositorio. El repositorio se describe en la siguiente sección.
 
## Estructura del repositorio

La estructura del repositorio es la siguiente:
 * **settings.py**: parámetros globales y creación de directorios en caso necesario. 
 * **preprocessor.py**: En este módulo se definen funciones que son necesarias para el procesamiento de los datos, necesarios para todos los apartados.
 * **punctuator_basic.py**: Funciones del Apartado 1 correspondientes a la puntuación básica.
 * **model_ngram.py**: Clase que define el modelo predictivo basado en N-gramas del Apartado 4.
 * **punctuator_ngram**: Funciones del Apartado 4 correspondientes a la puntuación basada en 4-gramas.
 * **evaluator.py**: Módulo que contiene las funciones para evaluar los módelos usados en el trabajo, esto es, los métodos correspondientes a los Apartados 2, 3 y 5.

En cuanto a los directorios tenemos:

    -root
        |-data  
        |  |-prepared
        |  |-preprocessed
        |  |-raw
        |-predicted
        |-punctuator2tf2
        |-zips

 * **data**: Contiene subdirectorios tando de datos del dataser original como procesados.
 * **raw**: Contiene los corpus de dataset originales.
 * **prepared**: Contiene archivos generados de preprocesamiento de los corpus (apartado 6)
 * **preprocessed**: Contiene archivos generados de preprocesamiento de los corpus vectorizados para el modelo de Ottokar Tilk y Tanel Alum (apartado 6).
 * **predicted**: Contiene archivos con predicciones (puntuaciones) realizadas por los modelos.
 * **punctuator2tf2**: Modelo de Ottokar Tilk y Tanel Alum (apartado 6), implementación en TensorFLow con modificaciones
 realizadas por mí para adaptalo a este trabajo.
 * **zips**: contiene el dataset y el modelo punctuator2tf2 comprimidos.
 
**NOTA**: A lo largo del trabajo voy a ir importando las funciones de cada archivo conforma hagan falta, recomiendo ir al código para consultarlas conforme se avanza en este documento.
