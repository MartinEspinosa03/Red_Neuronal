import os
import cv2
import np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

classes = [
    'Cadena', 'Calsetas', 'Camisa', 'Camiseta',
    'Corbata', 'Gorra', 'Lentes_sol',
    'Manga_Larga', 'Pans', 'Pantalon', 'Playera', 'Pulsera', 'Tenis'
]

num_classes = len(classes)  
img_rows, img_cols = 224, 224  

def load_data():
    data = []
    target = []
    for index, clase in enumerate(classes):
        folder_path = os.path.join(r'C:\8Cuatri\RN_ACC\data', clase) 
        print(f"Looking in: {folder_path}") 
        if not os.path.exists(folder_path):
            print(f"Folder does not exist: {folder_path}")
            continue
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            print(f"Loading image: {img_path}")  
            try:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Carga de la imagen en color
                if img is not None:
                    img = cv2.resize(img, (img_rows, img_cols))  # Redimensionar la imagen
                    data.append(np.array(img))  # Agregar imagen al dataset
                    target.append(index)  # Agregar clase correspondiente
                else:
                    print(f"Failed to load image: {img_path}")
            except Exception as e:
                print(f"Error reading file {img_path}: {e}")
                continue
    if data:
        data = np.array(data)
        data = data.astype('float32') / 255.0  # Normalización de las imágenes
        target = to_categorical(np.array(target), num_classes)  # Codificación one-hot de las etiquetas
    else:
        print("No images found.")
    return data, target

data, target = load_data()

# División de los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Generador de datos de imágenes con aumentación
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Carga del modelo base VGG16 preentrenado
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

# Añadir capas adicionales para la clasificación
x = base_model.output
x = Flatten()(x)  # Capa Flatten para aplanar la salida de VGG16
x = Dense(128, activation='relu')(x)  # Capa Dense con 128 neuronas y activación ReLU
x = Dropout(0.5)(x)  # Capa Dropout con un 50% de probabilidad de desconexión de neuronas
predictions = Dense(num_classes, activation='softmax')(x)  # Capa de salida con activación softmax y tantas neuronas como clases
#13 neuronas de salida una por cada clase

# Crear el modelo final combinando el modelo base y las capas adicionales
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar las capas del modelo base (VGG16) para no entrenarlas
for layer in base_model.layers:
    layer.trainable = False

# Compilación del modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                    validation_data=(X_test, y_test), 
                    epochs=5)

# Guardar el modelo entrenado
model_path = r'C:\8Cuatri\RN_ACC\models\modelo.h5'
model.save(model_path)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Generar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Guardar la matriz de confusión en un archivo
confusion_matrix_path = r'C:\8Cuatri\RN_ACC\graficas\confusion_matrix.png'
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(confusion_matrix_path)

# Graficar la pérdida durante el entrenamiento y la validación
training_loss_path = r'C:\8Cuatri\RN_ACC\graficas\training_loss.png'
plt.figure()
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento y la validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.savefig(training_loss_path)
plt.show()
# - El modelo utiliza VGG16 como base con sus capas congeladas, lo que significa que no se entrenarán.
# - La red tiene 16 capas, incluyendo 13 capas convolucionales y 3 capas totalmente conectadas.
# - Se añaden capas personalizadas: Flatten, Dense(128 neuronas) con activación ReLU, Dropout y una capa Dense de salida con tantas neuronas como clases (13).
# - El modelo se entrena con imágenes de 224x224 y se normalizan dividiendo por 255.
# - El optimizador utilizado es Adam y la función de pérdida es categorical_crossentropy.
# - La precisión se evalúa utilizando la métrica 'accuracy'.
# - La matriz de confusión y las gráficas de la pérdida se guardan como archivos para análisis posterior.