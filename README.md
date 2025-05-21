# Red Neuronal Artificial para Clasificación de Dígitos MNIST

Este repositorio contiene la implementación de una Red Neuronal Convolucional (CNN) utilizando PyTorch para clasificar dígitos escritos a mano del conjunto de datos MNIST. El proyecto fue desarrollado como parte del curso **EL5857 Aprendizaje Automático** en la Escuela de Ingeniería Electrónica del Instituto Tecnológico de Costa Rica, I Semestre 2025.

---

## Contenido del Repositorio

- `ann.py`: Implementación de la red neuronal convolucional (CNN)  
- `classifier.py`: Interfaz base para todos los clasificadores  
- `train_ann.py`: Script para entrenar la CNN con los parámetros óptimos  
- `test_ann.py`: Script para evaluar el modelo entrenado en conjuntos de datos de prueba  
- `mnist_cnn_best.pt`: Modelo pre-entrenado con la configuración óptima  
- `best_model_config_focused.json`: Configuración óptima para la arquitectura de la red  
- `Data.npy`: Datos de ejemplo para pruebas  
- `requirements.txt`: Dependencias necesarias para ejecutar el proyecto  

---

## Requisitos

Para ejecutar este proyecto, necesitarás instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

Las principales dependencias incluyen:

- PyTorch  
- NumPy  
- Matplotlib  
- scikit-learn  
- Weights & Biases (para seguimiento de experimentos)  

---

## Cómo Ejecutar

### Entrenar la Red Neuronal

Para entrenar la red neuronal con la configuración óptima:

```bash
python train_ann.py
```

Este script cargará la configuración desde `best_model_config_focused.json` y entrenará la CNN en el conjunto de datos MNIST. El modelo entrenado se guardará como `mnist_cnn_best.pt`.

### Evaluar el Modelo

Para evaluar el modelo en el conjunto de prueba MNIST:

```bash
python test_ann.py --dataset mnist
```

Para evaluar en datos escritos a mano por estudiantes:

```bash
python test_ann.py --dataset students
```

---

## Arquitectura

La CNN implementada tiene las siguientes características:

- **Capas convolucionales**: Configuradas como `[32, 64, 96]` filtros  
- **Kernels**: Tamaños `[3, 5, 7]`  
- **Capas fully connected**: `[512, 256]`  
- **Regularización**: Dropout (tasa ~0.58)  
- **Batch normalization**: Activada  
- **Función de activación**: ReLU  

La arquitectura se puede configurar modificando el archivo `best_model_config_focused.json` o directamente en el código.

---

## Resultados

El modelo entrenado alcanza una **precisión superior al 98%** en el conjunto de prueba MNIST.  
Los resultados detallados incluyendo **matrices de confusión**, **precisión** y **exhaustividad por clase** se generan automáticamente al ejecutar el script de evaluación.

Se puede visualizar el rendimiento del modelo y seguir el experimento a través de **Weights & Biases**.

---

## Sobre el Proyecto

Este proyecto fue desarrollado como parte del curso de Aprendizaje Automático, donde se exploraron varias técnicas de aprendizaje supervisado para el reconocimiento de dígitos. La CNN implementada representa la arquitectura optimizada tras múltiples experimentos de selección de hiperparámetros y análisis de optimalidad Pareto.

---

## Contacto

Para más información o preguntas sobre este proyecto, por favor contactar a:

**Elías Miranda Cedeño**
(elias27mc@gmail.com)

---

> **Nota:** Este repositorio es parte de un trabajo académico y está destinado a fines educativos.
