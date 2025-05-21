# Copyright (C) 2024-2025 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2025
# Proyecto 1

class Classifier:
    '''Interface class for a classifier.

       All classifiers to be used by the digit predictor should inherit from
       this interface class.
    '''

    def __init__(self,name):
        '''
        Classes implementing this interface must specify a classifier name
        that will be used, among others, as the name of the buttons of
        the graphic interface to invoque a particular classifier.
        '''

        self.name = name
    
    def load(self,filename):
        # Load the classifier model
        raise NotImplementedError

    def save(self,filename):
        # Save the classifier model
        raise NotImplementedError
    
    def predict(self, image):
        # Predict the class label for the given image
        # The image passed will always be a 28x28 pixel numpy array
        # The expected returned value will be the numerical id of the predicted
        # class, or None if an error occurred.
        
        raise NotImplementedError
