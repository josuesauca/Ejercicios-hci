import tkinter as tk
import cv2
import numpy as np
import os
from tkinter import messagebox, filedialog

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw, ImageGrab
import torchvision.transforms as transforms

#Librerias para mnist
import torch
import torch.nn as nn
from torchvision import datasets, transforms

#Neurona para realizar el entrenamiento respectivo
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if not os.path.isfile('model.pth'):
    # Descargar el conjunto de datos y entrenar el modelo
	train_dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

	model = NeuralNetwork().to(device)

	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

	def train(dataloader, model, loss_fn, optimizer):
	    size = len(dataloader.dataset)
	    model.train()
	    for batch, (X, y) in enumerate(dataloader):
	        X, y = X.to(device), y.to(device)

	        # Compute prediction error
	        pred = model(X)
	        loss = loss_fn(pred, y)

	        # Backpropagation
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()

	        if batch % 100 == 0:
	            loss, current = loss.item(), batch * len(X)
	            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

	epochs = 16
	for t in range(epochs):
	    print(f"Epoch {t+1}\n-------------------------------")
	    train(train_loader, model, loss_fn, optimizer)
	print("Done!")

	model.state_dict()
	torch.save(model.state_dict(), "model.pth")

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

def capture_drawing():
    if len(points_list) > 0:
        x_coords, y_coords = zip(*points_list)
        min_x, min_y = min(x_coords), min(y_coords)
        max_x, max_y = max(x_coords), max(y_coords)
        width = max_x - min_x
        height = max_y - min_y
        image = Image.new('L', (width+40, height+40), 'white')  # aumentar el tamaño de la imagen en 40 píxeles en cada dimensión
        draw = ImageDraw.Draw(image)
        for i in range(len(points_list)-1):
            x1, y1 = points_list[i]
            x2, y2 = points_list[i+1]
            draw.line((x1 - min_x+20, y1 - min_y+20, x2 - min_x+20, y2 - min_y+20), fill='black', width=10)  # agregar un desplazamiento de 20 píxeles a cada punto dibujado
        filename = 'drawing.jpg'
        image.save(os.path.join(os.getcwd(), filename), dpi=(300, 300))

        verificar_modelo(filename)


def select_image():
    file_path = filedialog.askopenfilename()
    verificar_modelo(file_path)

def verificar_modelo(file):
	test_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

	# Format Image
	img_resized = cv2.resize(test_image, (28, 28), interpolation=cv2.INTER_LINEAR)
	img_resized = cv2.bitwise_not(img_resized)

	# Resize the image to 28x28
	transform = ToTensor()

	img_tensor = transform(img_resized)
	img_tensor = img_tensor.unsqueeze(0)

	# Load the model
	model = NeuralNetwork()
	model.load_state_dict(torch.load("model.pth"))

	# Evaluate the model
	model.eval()
	with torch.no_grad():
	    output = model(img_tensor)
	    _, predicted = torch.max(output, 1)
	messagebox.showinfo("Número ingresado", f"El número ingresado es: {predicted.item()}")

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

clear_button = tk.Button(window, text="Limpiar Pantalla", command=clear_canvas)
clear_button.place(relx=0.75, rely=1.0, anchor='s')

capture_button = tk.Button(window, text="Identificar Imagen", command=capture_drawing)
capture_button.place(relx=0.25, rely=1.0, anchor='s')

select_button = tk.Button(window, text="Seleccionar Imagen", command=select_image)
select_button.place(relx=0.50, rely=1.0, anchor='s')

window.mainloop()
