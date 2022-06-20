import settings
from preprocessor import *

class ModelNgram():

    """
    Esta clase define el modelo predictivo basado en N-gramas del Apartado 4. Aunque en el trabajo se especifica explícitamente
    que sean 4gramas, la generalización es inmediata.

    Este modelo no requiere de la tokenización de los signos de puntuación por lo que puede trabajar con los datos
    en crudo.
    """
    # Definimos por defecto 4grams
    def __init__(self, N = 4):
        assert N > 1, 'N must be greater or equal 2' 
        self.N = N
        self.punct_marks = settings.PUNCT_MARKS
        self.mayus_marks = settings.MAYUS_MARKS
        self.minus = '<minus>'
        self.mayus = '<mayus>'
        # Diccionario con una clave para cada operación y valor un diccionario donde cada clave es una terna posible 
        # y su valor es el número de veces que aparece dicha terna seguida de la operación
        self.counts_dict = dict()
        # Diccionario donde cada clave es una terna y el valor es la operación predicha entrenada por el modelo
        # deducida del argumento del máximo en el diccionario counts_dict.
        self.trained_dict = dict()
        self.trained = False
                
    def entrena(self, train_file_path):
        """
        Entrena el modelo dada la ruta al corpus de entrenamiento 'train_file_path'
        """
        # Conjunto de tuplas vistas
        tuplas = set()
        
        # Inicializamos los subdiccionarios de cada operación
        self.counts_dict[self.mayus]= {}
        self.counts_dict[self.minus]={}
        for s in self.punct_marks:
            self.counts_dict[s] = {}
        
        # Recorremos el archivo de entrenamiento
        with open(train_file_path, 'r', encoding='utf-8') as train:
            train_lines = train.readlines()
            for line in train_lines:
                # Eliminamos el espacio final y retorno de carro del final de las líneas
                line = line.rstrip(' \n')
                # Obtenemos los 4gramas de la línea
                ngs = ngrams(line,self.N)
                for ng in ngs:
                    # obtenemos la operación correspondiente a partir del último elemento del 4grama
                    last = ng[-1]
                    op = last if last in self.punct_marks else self.mayus if last.isupper() else self.minus 
                    # Construimos la N-1 tupla  
                    tupla =  tuple([x.lower() for x in list(ng)[:-1]]) if len(ng) > 2 else ng[0]
                    tuplas.add(tupla)
                    # Sumamos una ocurrencia a la entrada correspondiente del diccionario
                    self.counts_dict[op][tupla]= self.counts_dict[op].get(tupla,0) + 1
        train.close()
        # Construimos el diccionario trained_model a partir de counts_dict
        for tupla in tuplas:
            self.trained_dict[tupla] = self.operacion_mas_probable(tupla)
        self.trained = True
        
    def operacion_mas_probable(self, tupla):  
        """
        Dada una (N-1) tupla, devuelve la operación más probablea partir del máximo de ocurrencias 
        consultando counts_dict    
        """
        v = 0
        prediction = 'NONE'
        for i in self.counts_dict:
            value = self.counts_dict[i].get(tupla,0) 
            if value > v:
                prediction = i
                v = value
        return prediction
    
    def predice(self,tupla):
        """
        Predice la operación dada una tupla consultando directamente el diccionario trained_model  
        """
        #Devolvemos excepción si modelo aún no entrenado
        self.check_entrenado()
        return self.trained_dict.get(tupla,'NONE')
    
    def check_entrenado(self):
         if not self.trained:
            raise NotTrainedModel(Exception("Model has not been trained yet"))
        
class NotTrainedModel(Exception): pass        