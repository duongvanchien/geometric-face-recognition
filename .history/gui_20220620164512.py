import PySimpleGUI as sg
import io, os
from PIL import Image
import cv2

import image_classification as ic


model=ic.initModel()

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
            result = ic.feature_extraction(values["-FILE-"])
            label = model.predict([result["features"]])[0]

            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
            origin_image = Image.fromarray(result["original_img"])
            origin_image.thumbnail((400,400))
            origin = io.BytesIO()
            origin_image.save(origin, format='PNG')
            
            edge_image = Image.fromarray(result["edge_img"])
            edge_image.thumbnail((400,400))
            edge = io.BytesIO()
            edge_image.save(edge, format='PNG')

            window["-O_IMAGE-"].update(data= origin.getvalue())
            window["-FIRST_TEXT-"].update("Original Image:")

            window["-E_IMAGE-"].update(data= edge.getvalue())
            window["-SECOND_TEXT-"].update("Edge Image:")

            window["-RESULT_TEXT-"].update("RESULT: " + str(label))
window.close()