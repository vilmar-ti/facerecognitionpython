import cv2
from keras.models import load_model
import numpy as np
import os
import time
from datetime import datetime
import shutil
import zipfile

# Carregar o classificador de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame da webcam
    ret, frame = cap.read()

    # Converter o frame para escala de cinza para detecção de faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces no frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Desenhar retângulos ao redor das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Mostrar o frame com as faces detectadas
    cv2.imshow('Video', frame)

    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


model = load_model('keras_model.h5',compile=False)
data = np.ndarray(shape=(1,224,224,3),dtype=np.float32)

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
