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

Este notebook contiene los apartados del 1 al 5 del trabajo, dejando el apartado 6 en el notebook TRABAJO_PLN_APARTADO_6_JULIAN_GALINDO_ALVAREZ_20091281E y el 7 en en documento TRABAJO_PLN_APARTADO_7_JULIAN_GALINDO_ALVAREZ_20091281E.pdf

Las funciones definidas en cada uno de los apartados se encuentran en los correspondientes archivos python de este directorio. En cualquier caso, este proyecto está disponible en el siguiente repositorio de github:

 * [github.com/julgalalv/PLN_Prediccion_de_puntuacion](https://github.com/julgalalv/PLN_Prediccion_de_puntuacion)
 
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

 * **data**: Contiene subdirectorios tando de datos del dataser original como procesados.
 * **raw**: Contiene los cospues de dataset originales.
 * **preprocessed**: Contiene archivos generados de preprocesamiento de los corpus (apartado 6)
 * **prepared**: Contiene archivos generados de preprocesamiento de los corpus preparados para el modelo de Ottokar Tilk y Tanel Alum (apartado 6).
 * **predicted**: Contiene archivos con predicciones (puntuaciones) realizadas por los modelos.
 * **punctuator2tf2**: Modelo de Ottokar Tilk y Tanel Alum (apartado 6), implementación en TensorFLow con modificaciones
 realizadas por mí para adaptalo a este trabajo.
 
**NOTA**: A lo largo del trabajo voy a ir importando las funciones de cada archivo conforma hagan falta, recomiendo ir al código para consultarlas conforme se avanza en este documento.
