import PySimpleGUI as sg
import io, os
from PIL import Image
import cv2

import image_classification as ic

file_types = [
    ("All files (*.*)", "*.*")
]

layout = [
    [
        [sg.Text(key="-FIRST_TEXT-", size=(50,1)),sg.Text(key="-SECOND_TEXT-")],
        [sg.Image(key="-O_IMAGE-"),sg.Image(key="-E_IMAGE-")]
    ],
    [
        sg.Text("Image File"),
        sg.Input(size=(25,1), key="-FILE-"),
        sg.FileBrowse(file_types=file_types),
        sg.Button("Submit Image")
    ],
    [
        sg.Text(key="-RESULT_TEXT-",font=("Arial, 15"))
    ]
]

window = sg.Window("Image Viewer", layout)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "Submit Image":
        filename = values["-FILE-"]
        if os.path.exists(filename):
            img= cv2.imread(values["-FILE-"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = Image.fromarray(img)
            image.thumbnail((400,400))
            bio = io.BytesIO()
            image.save(bio, format='PNG')
            window["-O_IMAGE-"].update(data= bio.getvalue())
            window["-FIRST_TEXT-"].update("Original Image:")
            window["-E_IMAGE-"].update(data= bio.getvalue())
            window["-SECOND_TEXT-"].update("Edge Image:")
            window["-RESULT_TEXT-"].update("RESULT:")
window.close()