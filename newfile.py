import os
import cv2
import numpy as np
import tensorflow as tf
import PySimpleGUI as sg
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Inicialize a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Inicialize o nome do arquivo do modelo
model_filename = "modelo_reconhecimento.h5"
test_data_filename = "test_data.npz"

# Função para criar o modelo
def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Crie ou carregue o modelo
if os.path.exists(model_filename):
    model = load_model(model_filename)
else:
    model = create_model()

# Carregue o classificador em cascata para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Defina o layout da interface gráfica
layout = [
    [sg.Image(filename="", key="-IMAGE-")],
    [sg.Text("Nome da Classe 1:"), sg.InputText(key="-CLASS_NAME_1-", size=(20, 1))],
    [sg.Button("Capturar Classe 1", key="-CAPTURE_CLASS_1-")],
    [sg.Text("Nome da Classe 2:"), sg.InputText(key="-CLASS_NAME_2-", size=(20, 1))],
    [sg.Button("Capturar Classe 2", key="-CAPTURE_CLASS_2-")],
    [sg.Button("Treinar Modelo", key="-TRAIN_MODEL-"), sg.Button("Prever Classe", key="-PREDICT_CLASS-"), sg.Button("Exibir Gráfico", key="-SHOW_GRAPH-")],
    [sg.Button("Exibir Resumo do Modelo", key="-SHOW_MODEL_SUMMARY-"), sg.Button("Avaliar Modelo", key="-EVALUATE_MODEL-"), sg.Button("Sair", key="-EXIT-")]
]

# Crie a janela
window = sg.Window("Interagindo com Modelo", layout, resizable=True, finalize=True)

# Lista para armazenar as imagens e rótulos de treinamento
all_images = []
all_labels = []

# Nomes das classes
class_names = ['', '']  # Inicializa com strings vazias

# Listas para armazenar o histórico de acurácia e loss durante o treinamento
accuracy_history = []
loss_history = []

# Conjunto de dados de teste
test_images = []
test_labels = []

while True:
    event, values = window.read(timeout=20)

    if event == sg.WIN_CLOSED or event == "-EXIT-":
        break

    # Capture quadro a quadro
    ret, frame = cap.read()

    # Converta a imagem para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecte rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenhe um retângulo ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Redimensione a imagem para 64x64 pixels
    gray = cv2.resize(gray, (64, 64))

    # Normalizar a imagem
    gray = gray / 255.0

    # Adicione uma dimensão extra
    gray = np.expand_dims(gray, axis=-1)

    # Atualize a imagem na interface gráfica
    imgbytes = cv2.imencode(".png", frame)[1].tobytes()
    window["-IMAGE-"].update(data=imgbytes)

    if event == "-CAPTURE_CLASS_1-":
        class_names[0] = values["-CLASS_NAME_1-"]  # Atualiza o nome da Classe 1
        # Capture a classe 1
        for _ in range(10):  # Capture 10 imagens
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))
            gray = gray / 255.0
            gray = np.expand_dims(gray, axis=-1)
            all_images.append(gray)
            all_labels.append([1, 0])

        sg.popup(f"Capturadas 10 imagens da {class_names[0]}")

    elif event == "-CAPTURE_CLASS_2-":
        class_names[1] = values["-CLASS_NAME_2-"]  # Atualiza o nome da Classe 2
        # Capture a classe 2
        for _ in range(10):  # Capture 10 imagens
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))
            gray = gray / 255.0
            gray = np.expand_dims(gray, axis=-1)
            all_images.append(gray)
            all_labels.append([0, 1])

        sg.popup(f"Capturadas 10 imagens da {class_names[1]}")

    elif event == "-TRAIN_MODEL-":
        # Converta as listas em arrays NumPy
        all_images_np = np.array(all_images)
        all_labels_np = np.array(all_labels)

        # Divida os dados em conjuntos de treinamento e teste
        train_images, test_images, train_labels, test_labels = train_test_split(
            all_images_np, all_labels_np, test_size=0.2, random_state=42
        )

        # Treine o modelo com os dados de treinamento e salve o histórico
        history = model.fit(train_images, train_labels, epochs=50)

        # Armazene a acurácia e loss do treinamento em cada época
        accuracy_history.extend(history.history['accuracy'])
        loss_history.extend(history.history['loss'])

        # Salve o modelo treinado
        model.save(model_filename)

        sg.popup("Treinamento concluído.")

    elif event == "-PREDICT_CLASS-":
        # Faça a previsão
        prediction = model.predict(np.array([gray]))
        predicted_class = class_names[np.argmax(prediction)]
        sg.popup(f'Classe prevista: {predicted_class}')

    # Exiba o gráfico quando clicar no botão de exibição do gráfico
    elif event == "-SHOW_GRAPH-":
        # Plote o gráfico de evolução da acurácia e loss
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(accuracy_history)
        plt.title('Evolução da Acurácia durante o Treinamento')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')

        plt.subplot(1, 2, 2)
        plt.plot(loss_history)
        plt.title('Evolução da Loss durante o Treinamento')
        plt.xlabel('Época')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.show()

    # Exiba o resumo do modelo quando clicar no botão correspondente
    elif event == "-SHOW_MODEL_SUMMARY-":
        # Imprima o resumo do modelo no console
        print("\nResumo do Modelo:")
        model.summary()
        print("\n")

    elif event == "-EVALUATE_MODEL-":
        if len(test_images) == 0 or len(test_labels) == 0:
            sg.popup("Carregue um conjunto de dados de teste antes de avaliar o modelo.")
        else:
            # Avalie o modelo no conjunto de dados de teste
            test_loss, test_accuracy = model.evaluate(test_images, test_labels)
            sg.popup(f"Acurácia no Conjunto de Teste: {test_accuracy * 100:.2f}%\nLoss no Conjunto de Teste: {test_loss}")

    # Carregue o conjunto de dados de teste quando clicar no botão correspondente
    elif event == "-LOAD_TEST_DATA-":
        if os.path.exists(test_data_filename):
            data = np.load(test_data_filename)
            test_images = data['test_images']
            test_labels = data['test_labels']
            sg.popup("Conjunto de Teste Carregado.")
        else:
            sg.popup("Nenhum conjunto de dados de teste encontrado. Capture dados e salve-os primeiro.")

# Quando tudo estiver pronto, libere a captura
cap.release()
cv2.destroyAllWindows()
window.close()
