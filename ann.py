# -*- coding: utf-8 -*-
"""
Implementacion de una Red Neuronal Convolucional (CNN) para MNIST.
Este archivo contiene la definicion de la clase CNN que implementa la interfaz Classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier import Classifier

class CNN(nn.Module, Classifier):
    '''
    Clasificador de Red Neuronal Convolucional.
    
    Esta clase define una CNN personalizable para reconocimiento de digitos MNIST.
    Hereda de nn.Module (PyTorch) y de la interfaz Classifier del proyecto.
    '''
    
    def __init__(self, filename=None, config=None):
        """
        Inicializa la CNN con los parametros dados o con valores predeterminados.
        
        Args:
            filename (str, optional): Ruta al archivo de modelo guardado. Default: None.
            config (dict, optional): Diccionario con la configuracion del modelo. Default: None.
        """
        nn.Module.__init__(self)
        Classifier.__init__(self, "CNN")
        
        # Configuracion default o la proporcionada
        if config is None:
            # Valores por defecto (pueden ser reemplazados por el proceso de seleccion)
            self.conv_channels = (32, 64, 96)
            self.kernel_sizes = (3, 5, 7)
            self.fc_units = (512, 256)
            self.dropout_rate = 0.5791545562018217
            self.use_batch_norm = True
            self.activation_name = 'relu'
        else:
            # Usar configuracion proporcionada y convertir listas a tuplas si es necesario
            self.conv_channels = tuple(config.get('conv_channels', (64, 128, 256)))
            self.kernel_sizes = tuple(config.get('kernel_sizes', (3, 5, 7)))
            self.fc_units = tuple(config.get('fc_units', (512,)))
            self.dropout_rate = config.get('dropout_rate', 0.5)
            self.use_batch_norm = config.get('use_batch_norm', True)
            self.activation_name = config.get('activation', 'relu')
        
        # Calcular padding para mantener el tama?o despu?s de cada convoluci?n
        # Para kernel_size k, padding = (k-1)//2 mantiene el tama?o
        padding1 = (self.kernel_sizes[0] - 1) // 2
        padding2 = (self.kernel_sizes[1] - 1) // 2
        padding3 = (self.kernel_sizes[2] - 1) // 2
        
        # Capas convolucionales con diferentes kernel sizes
        self.conv1 = nn.Conv2d(1, self.conv_channels[0], kernel_size=self.kernel_sizes[0], stride=1, padding=padding1)
        self.bn1 = nn.BatchNorm2d(self.conv_channels[0]) if self.use_batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(self.conv_channels[0], self.conv_channels[1], kernel_size=self.kernel_sizes[1], stride=1, padding=padding2)
        self.bn2 = nn.BatchNorm2d(self.conv_channels[1]) if self.use_batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(self.conv_channels[1], self.conv_channels[2], kernel_size=self.kernel_sizes[2], stride=1, padding=padding3)
        self.bn3 = nn.BatchNorm2d(self.conv_channels[2]) if self.use_batch_norm else nn.Identity()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Construir capas fully connected dinamicamente
        # Tama?o de entrada despues de conv layers: conv_channels[2] * 3 * 3
        fc_input_size = self.conv_channels[2] * 3 * 3
        
        # Listas para almacenar las capas FC
        self.fc_layers = nn.ModuleList()
        self.bn_fc_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # Crear capas intermedias
        prev_size = fc_input_size
        for fc_size in self.fc_units:
            # Capa linear
            self.fc_layers.append(nn.Linear(prev_size, fc_size))
            # Batch normalization (si esta habilitada)
            if self.use_batch_norm:
                self.bn_fc_layers.append(nn.BatchNorm1d(fc_size))
            else:
                self.bn_fc_layers.append(nn.Identity())
            # Dropout
            self.dropout_layers.append(nn.Dropout(self.dropout_rate))
            prev_size = fc_size
        
        # Capa de salida (siempre 10 clases para MNIST)
        self.output_layer = nn.Linear(prev_size, 10)
        
        # Dispositivo (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cargar modelo si se proporciona un archivo
        if filename is not None:
            self.load(filename)
        
        # Mover el modelo al dispositivo adecuado
        self.to(self.device)
    
    def activation(self, x):
        """
        Aplica la funcion de activacion segun la configuracion.
        
        Args:
            x (torch.Tensor): Tensor de entrada
            
        Returns:
            torch.Tensor: Tensor con la activacion aplicada
        """
        if self.activation_name == 'relu':
            return F.relu(x)
        elif self.activation_name == 'leaky_relu':
            return F.leaky_relu(x, 0.1)
        elif self.activation_name == 'prelu':
            # Nota: para PReLU seria mejor usar nn.PReLU(), pero aqui usamos F.leaky_relu por simplicidad
            return F.leaky_relu(x, 0.01)
        elif self.activation_name == 'elu':
            return F.elu(x)
        return F.relu(x)  # Default
    
    def forward(self, x):
        """
        Realiza la pasada hacia adelante (forward) del modelo.
        
        Args:
            x (torch.Tensor): Tensor de entrada con forma [batch_size, channels, height, width]
                              o [height, width] para una sola imagen
                              
        Returns:
            torch.Tensor: Logits para cada una de las 10 clases de digitos
        """
        # Asegurar que x tiene la forma correcta [batch_size, 1, 28, 28]
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
            
        # Capas convolucionales con activacion y MaxPooling
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Aplanar antes de las capas fully connected
        x = x.view(-1, self.conv_channels[2] * 3 * 3)
        
        # Pasar a traves de todas las capas fully connected
        for i, (fc_layer, bn_layer, dropout_layer) in enumerate(zip(self.fc_layers, self.bn_fc_layers, self.dropout_layers)):
            x = fc_layer(x)
            x = bn_layer(x)
            x = self.activation(x)
            x = dropout_layer(x)
        
        # Capa de salida (sin activacion ni dropout)
        x = self.output_layer(x)
        
        return F.log_softmax(x, dim=1)
    
    def load(self, filename):
        """
        Carga los pesos del modelo desde un archivo.
        
        Args:
            filename (str): Ruta al archivo del modelo guardado
        """
        self.load_state_dict(torch.load(filename, map_location=self.device))
        self.to(self.device)
        self.eval()
    
    def save(self, filename):
        """
        Guarda los pesos del modelo en un archivo.
        
        Args:
            filename (str): Ruta donde guardar el modelo
        """
        self.to('cpu')  # Es buena practica guardar en modo CPU
        torch.save(self.state_dict(), filename)
        self.to(self.device)  # Volver al dispositivo original
    
    def predict(self, image):
        """
        Predice el digito para una imagen dada.
        
        Args:
            image (numpy.ndarray): Imagen de 28x28 en formato numpy
            
        Returns:
            int: Digito predicho (0-9)
        """
        # Convertir la imagen numpy a tensor PyTorch
        image_tensor = torch.from_numpy(image).float().to(self.device)
        
        # Preparar la imagen para la CNN (anadir dimensiones de batch y canal)
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
        
        # Evaluacion del modelo
        self.eval()
        with torch.no_grad():
            output = self(image_tensor)
            pred = output.argmax(dim=1)
            
        return pred.item()
