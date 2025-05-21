# -*- coding: utf-8 -*-
"""
Script para entrenar la red neuronal con los hiperparametros optimizados.
Este script carga la configuracion optima encontrada y entrena el modelo final.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import json
import wandb
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from ann import CNN
from Data_Augmentation import transform_train, transform_test, configurar_aumentacion

def train_model(config, usar_aumento=True):
    """
    Entrena el modelo final con la configuracion optima encontrada.
    
    Args:
        config (dict): Diccionario con la configuracion del modelo
        usar_aumento (bool): Si se debe usar aumento de datos
        
    Returns:
        tuple: (modelo entrenado, precision, exhaustividad)
    """
    # Configurar aumento de datos
    configurar_aumentacion(usar_aumento)
    
    # Inicializar wandb
    wandb.init(
        project="mnist-cnn-mejores-modelos",
        name=f"final-training-{'con' if usar_aumento else 'sin'}-aumento",
        config=config
    )
    
    # Seleccionar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Crear modelo con la configuracion optima
    model = CNN(config=config['config']).to(device)
    
    # Preparar datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform_test)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['config']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['config']['batch_size'], shuffle=False)
    
    # Criterio
    criterion = nn.CrossEntropyLoss()
    
    # Optimizador
    optimizer_name = config['config']['optimizer_name']
    lr = config['config']['learning_rate']
    weight_decay = config['config']['weight_decay']
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:  # rmsprop
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Entrenamiento
    epochs = 30  # Mas epocas para el modelo final
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        # Modo entrenamiento
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoca {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        train_loss /= len(train_loader)
        
        # Modo evaluacion
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        # Log en wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_accuracy': accuracy
        })
        
        print(f'Epoca {epoch}: Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Guardar mejor modelo
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            model.save("mnist_cnn_best.pt")
            print(f"Guardado nuevo mejor modelo con test loss: {test_loss:.4f}")
        
        # Actualizar learning rate
        scheduler.step(test_loss)
    
    # Guardar modelo final
    model.save("mnist_cnn_final.pt")
    
    # Evaluar modelo final
    model.load("mnist_cnn_best.pt")
    
    # Calcular metricas finales y matriz de confusion
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Matriz de confusion
    cm = confusion_matrix(all_targets, all_preds)
    
    # Calcular precision y recall
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    
    # Precision y recall por clase
    precision_per_class = precision_score(all_targets, all_preds, average=None)
    recall_per_class = recall_score(all_targets, all_preds, average=None)
    
    # Crear y guardar visualizacion de la matriz de confusion
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusion')
    plt.colorbar()
    plt.xticks(range(10), range(10))
    plt.yticks(range(10), range(10))
    plt.xlabel('Prediccion')
    plt.ylabel('Verdadero')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Log metricas finales en wandb
    wandb.log({
        'final_test_loss': best_test_loss,
        'final_precision': precision,
        'final_recall': recall,
        'confusion_matrix': wandb.Image('confusion_matrix.png')
    })
    
    # Log de metrics por clase
    for i in range(10):
        wandb.log({
            f'precision_digit_{i}': precision_per_class[i],
            f'recall_digit_{i}': recall_per_class[i]
        })
    
    print(f"\nResultados Finales:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Exhaustividad: {recall:.4f}")
    
    wandb.finish()
    return model, precision, recall

if __name__ == "__main__":
    # Cargar la configuracion del mejor modelo
    try:
        with open('best_model_config_focused.json', 'r') as f:
            best_config = json.load(f)
        
        print("Entrenando con la configuracion optima encontrada:")
        train_model(best_config, usar_aumento=True)
        
    except FileNotFoundError:
        print("No se encontro el archivo de configuracion optima.")
        print("Ejecutando con configuracion predeterminada...")
        
        default_config = {
            'config': {
                'conv_channels': (32, 64, 128),
                'fc_units': (256, 128),
                'dropout_rate': 0.5,
                'use_batch_norm': True,
                'activation': 'relu',
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'optimizer_name': 'adam',
                'batch_size': 64
            }
        }
        
        train_model(default_config, usar_aumento=True)


