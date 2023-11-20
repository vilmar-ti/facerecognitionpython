import cv2
import numpy as np
import tensorflow as tf
import os
import PySimpleGUI as sg

print("TensorFlow version:", tf.__version__)

# Carregando o modelo de detecção de rostos pré-treinado do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # Use 0 para a câmera padrão
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(height, width, 1)),  # Corrigindo a forma de entrada
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(25, activation="sigmoid", name="perceptron_1"),
    tf.keras.layers.Dense(25, activation="sigmoid", name="perceptron_2"),
    tf.keras.layers.Dense(2, activation="softmax", name="output")
])

model.build()

model.compile(optimizer=tf.optimizers.Adagrad(lr=0.01),
              loss={'output': 'mse'},
              metrics={'output': 'accuracy'})

f = []
images = []

EPOCHS = 100

# Diretório onde as imagens serão salvas
output_directory = "captured_images"
os.makedirs(output_directory, exist_ok=True)

# Caminho para salvar o modelo treinado
model_save_path = "face_recognition_model.h5"

# Layout da interface gráfica
layout = [
    [sg.Image(filename="", key="-IMAGE-")],
    [sg.Button("Exit"), sg.Button("Capture Class 1", key="-CLASS1-"), sg.Button("Capture Class 2", key="-CLASS2-"),
     sg.Button("Train Model", key="-TRAIN-"), sg.Button("Predict", key="-PREDICT-"),
     sg.Button("Load Model", key="-LOAD_MODEL-")]
]

window = sg.Window("Face Recognition System", layout, resizable=True)

grab_image = False

# Carregar o modelo treinado, se existir
if os.path.exists(model_save_path):
    model = tf.keras.models.load_model(model_save_path)
    print(f"Model loaded from {model_save_path}")

while True:
    event, values = window.read(timeout=20)

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if event in (sg.WIN_CLOSED, "Exit"):
        break
    elif event == "-CLASS1-":
        initial_class = np.array([1.0, 0.0])
        grab_image = True
        print("Classe 1")
    elif event == "-CLASS2-":
        initial_class = np.array([0.0, 1.0])
        grab_image = True
        print("Classe 2")
    elif event == "-TRAIN-":
        model.fit(np.array(images), np.array(f), epochs=EPOCHS)

        # Salvar o modelo treinado
        model.save(model_save_path)
        print(f"Model saved at {model_save_path}")

    elif event == "-PREDICT-":
        # Restante do código de previsão
        preprocessed_frame = gray / 255.0  # Normalização
        frame_input = np.expand_dims(np.expand_dims(preprocessed_frame, axis=0), axis=-1)  # Adicionando dimensões do canal e do lote
        predict = model.predict(frame_input)
        predict_label = list(np.where(predict[0] >= 0.5, 1, 0))
        max_value = max(predict_label)
        max_index = predict_label.index(max_value)
        labels = ["Classe1", "Classe2"]
        print("Predict Label", labels[max_index])
        print("predict", predict)
    
    elif event == "-LOAD_MODEL-":
        # Diálogo para selecionar o arquivo do modelo
        model_file_path = sg.popup_get_file("Select a model file", file_types=(("Model Files", "*.h5"),))

        if model_file_path:
        # Carregar o modelo selecionado
            model = tf.keras.models.load_model(model_file_path)
            print(f"Model loaded from {model_file_path}")

    if grab_image:
        preprocessed_frame = gray / 255.0  # Normalização
        frame_input = np.expand_dims(np.expand_dims(preprocessed_frame, axis=0), axis=-1)  # Adicionando dimensões do canal e do lote

        # Salve a imagem capturada no diretório
        image_filename = f"{output_directory}/captured_image_{len(images)}.png"
        cv2.imwrite(image_filename, gray)

        images.append(frame_input)
        f.append(initial_class)
        grab_image = False

    window["-IMAGE-"].update(data=cv2.imencode(".png", frame)[1].tobytes())

window.close()
cap.release()
cv2.destroyAllWindows()
