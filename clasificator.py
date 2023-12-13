import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split, KFold
# from skimage.transform import resize
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Ruta al directorio que contiene los archivos de audio
audio_dir = "/Users/guillermogarciaciai/downloads/Data/genres_original"

# Obtén la lista de géneros disponibles en el conjunto de datos
genres = os.listdir(audio_dir)

# Crear listas para almacenar espectrogramas y etiquetas
spectrograms = []
labels = []

# Configurar las dimensiones de los espectrogramas
spectrogram_height = 128
spectrogram_width = 128

# Procesar los archivos de audio
for genre in genres:
    genre_dir = os.path.join(audio_dir, genre)
    if genre_dir == '/Users/guillermogarciaciai/downloads/Data/genres_original/.DS_Store':
        continue
    for filename in os.listdir(genre_dir):
        file_path = os.path.join(genre_dir, filename)
        # Verificar si es un directorio
        if os.path.isdir(file_path):
            continue
        # Cargar el archivo de audio con librosa
        try:
            y, sr = librosa.load(file_path, duration=30, sr=None)  # Duración de 30 segundos
        except Exception:
            continue
        # Generar el espectrograma
        #spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        # Cambiar el tamaño del espectrograma a las dimensiones deseadas
        #spectrogram = resize(spectrogram, (spectrogram_height, spectrogram_width), anti_aliasing=True)
        #spectrograms.append(spectrogram)
        #labels.append(genre)

# Convertir las listas en matrices NumPy
spectrograms = np.array(spectrograms)
labels = np.array(labels)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(spectrograms, labels, test_size=0.2, random_state=42)

# Codificar las etiquetas en números (por ejemplo, usando LabelEncoder)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Definir el número de géneros en el conjunto de datos
num_genres = len(np.unique(y_train_encoded))

# Definir y compilar el modelo CNN con regularización L2 y capa de abandono
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(spectrogram_height, spectrogram_width, 1), kernel_regularizer=l2(0.01)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),  # Capa de abandono con tasa de abandono del 50%
    keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    keras.layers.Dense(num_genres, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Validación cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in kf.split(X_train):
    X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
    y_cv_train, y_cv_test = y_train_encoded[train_index], y_train_encoded[test_index]

    model.fit(X_cv_train, y_cv_train, epochs=10, verbose=0)

    y_cv_pred = model.predict(X_cv_test)
    y_cv_pred = np.argmax(y_cv_pred, axis=1)
    accuracy = accuracy_score(y_cv_test, y_cv_pred)
    accuracies.append(accuracy)
    print("Precisión en esta división:", accuracy)

average_accuracy = sum(accuracies) / len(accuracies)
print("Precisión promedio en validación cruzada:", average_accuracy)

# Entrenar el modelo final en todos los datos de entrenamiento
model.fit(X_train, y_train_encoded, epochs=10)

# Clasificar un archivo de audio
audio_path = f"/Users/guillermogarciaciai/Downloads/Eric Clapton - Groaning.wav"  # Reemplaza con la ruta de tu archivo de audio
y, sr = librosa.load(audio_path, duration=30, sr=None)  # Duración de 30 segundos

# Generar el espectrograma y cambiar el tamaño (como hiciste en el proceso de entrenamiento)
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
spectrogram = resize(spectrogram, (spectrogram_height, spectrogram_width), anti_aliasing=True)

# Reformatear el espectrograma
input_data = spectrogram.reshape(1, spectrogram_height, spectrogram_width, 1).astype('float32')

# Realizar una predicción
predicted_probabilities = model.predict(input_data)

# Decodificar la predicción para obtener el género musical
predicted_genre_index = np.argmax(predicted_probabilities)
predicted_genre = label_encoder.inverse_transform([predicted_genre_index])

print("El género musical predicho es:", predicted_genre[0])
