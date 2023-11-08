import tkinter as tk
import cv2
import numpy as np

import os
from PIL import Image, ImageDraw, ImageGrab

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

'''
def capture_drawing():
    if len(points_list) > 0:
        filename = 'drawing.png'
        image = ImageGrab.grab(bbox=(canvas.winfo_rootx(), canvas.winfo_rooty(), canvas.winfo_rootx() + canvas.winfo_width(), canvas.winfo_rooty() + canvas.winfo_height()))
        image.show()
	
        image.save(os.path.join(os.getcwd(), filename))
        
        print('Drawing saved as', filename)
'''

def capture_drawing():
    if len(points_list) > 0:
        filename = 'drawing.png'
        image = ImageGrab.grab(bbox=(canvas.winfo_rootx(), canvas.winfo_rooty(), canvas.winfo_rootx() + canvas.winfo_width(), canvas.winfo_rooty() + canvas.winfo_height()))
        image.save(os.path.join(os.getcwd(), filename))
        
        # Convert image to grayscale
        gray_image = image.convert('L')
        
        # Apply threshold to create binary image
        threshold = 128
        binary_image = gray_image.point(lambda x: 0 if x < threshold else 255, '1')
        print(binary_image, 'hola ')
        # Find contours of number

        '''
        contours = cv2.findContours(np.array(binary_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(contours) > 0:
            # Get bounding box of largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Crop image to bounding box
            number_image = image.crop((x, y, x+w, y+h))
            number_image.show()
            number_image.save('number.png')
        '''

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

capture_button = tk.Button(window, text="Capture", command=capture_drawing)
capture_button.place(relx=0.25, rely=1.0, anchor='s')

window.mainloop()