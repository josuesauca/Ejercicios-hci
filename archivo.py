import tkinter as tk
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageGrab

#Libreria para reconocer texto de imagenes
import aspose.ocr as ocr

#Librerias para mnist

import torch
import torch.nn as nn
from torchvision import datasets, transforms


mnist = datasets.MNIST(root='.', train=True, download=True)


points_list = []

def draw_on_canvas(event):
    brush_size = 20
    x = event.x
    y = event.y
    canvas.create_oval((x -brush_size/2 , y - brush_size/2, x + brush_size/2, y + brush_size/2), fill='black')
    points_list.append((x , y))

def clear_canvas():
    canvas.delete('all')
    points_list.clear()

# create a button to save the canvas object
def save_canvas():
    canvas.postscript(file="drawing.eps", colormode="color")
    img = Image.open("drawing.eps")
    img.save("drawing.png", "png")

def capture_drawing():
    if len(points_list) > 0:
        x_coords, y_coords = zip(*points_list)
        min_x, min_y = min(x_coords), min(y_coords)
        max_x, max_y = max(x_coords), max(y_coords)
        width = max_x - min_x
        height = max_y - min_y
        image = Image.new('L', (width, height), 'white') 
        draw = ImageDraw.Draw(image)
        for i in range(len(points_list)-1):
            x1, y1 = points_list[i]
            x2, y2 = points_list[i+1]
            draw.line((x1 - min_x, y1 - min_y, x2 - min_x, y2 - min_y), fill='black', width=10)  # Draw thicker white lines
        filename = 'drawing.png'
        #image = image.resize((28, 28))
        image.save(os.path.join(os.getcwd(), filename), dpi=(300, 300))
        #print('Drawing saved as', filename)
        image.show()
def reconocer_texto():

	for i in range(len(mnist)):
	    if (mnist[i][1] < 10 and len(images) < num_images and bandera) :
	        #images.append(mnist[i][0]) sirve para obtener la imagen
	        #labels.append(mnist[i][1]) sirve para obtener el nombre en entero
	        if mnist[i][1] == numero:
	           num_obtenido = mnist[i][1]
	           bandera = False

	print('Es el numero: ',num_obtenido)


window = tk.Tk()
window.title('Canvas')

# Obtener el ancho y alto de la pantalla
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Obtener el ancho y alto de la ventana
window_width = 600
window_height = 400

# Calcular las coordenadas x e y de la ventana para que se muestre en el centro de la pantalla
x = int((screen_width/2) - (window_width/2))
y = int((screen_height/2) - (window_height/2))

# Establecer la geometría de la ventana con las coordenadas calculadas
window.geometry('{}x{}+{}+{}'.format(window_width, window_height, x, y))

canvas = tk.Canvas(window, background="white", width=window_width, height=window_height)
canvas.place(relx=0.5, rely=0.5, anchor='center')
canvas.bind('<B1-Motion>', draw_on_canvas)

clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.place(relx=0.75, rely=1.0, anchor='s')

#capture_button = tk.Button(window, text="Capture", command=reconocer_texto)
capture_button = tk.Button(window, text="Capture", command=capture_drawing)
capture_button.place(relx=0.25, rely=1.0, anchor='s')


window.mainloop()
