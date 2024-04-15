import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model

# Cargar el modelo
model = load_model('modelo.h5')
classes = ['cuchara', 'Cuchara_de_servir', 'Cucharon_escurridor', 'Cuchillo', 'Escurridor', 'Espatula', 'Espatula_revolvedora', 'Olla', 'Pelador', 'Prensador_de_ajo', 'Sarten', 'Tabla_de_cortar', 'Tapa_de_olla', 'Tenedor', 'Vasija']

def load_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    image = np.array(image)
    image = image.reshape(1, 64, 64, 1)
    return image

def predict_class(image):
    image = load_image(image)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    return classes[predicted_class]

def video_stream():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label_img.imgtk = imgtk
    label_img.configure(image=imgtk)
    label_text.config(text = f'Predicted Class: {predict_class(frame)}')
    label_img.after(10, video_stream) 

root = tk.Tk()

label_img = tk.Label(root)
label_img.pack(pady=10)

label_text = tk.Label(root, text='')
label_text.pack()

cap = cv2.VideoCapture(0)

video_stream()
root.mainloop()