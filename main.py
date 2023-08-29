# pip install matplotlib pytesseract opencv-python tesseract-ocr
from flask import Flask, render_template, request, redirect, url_for, session
import subprocess
import matplotlib.pyplot as plt
import pytesseract
import cv2

app = Flask(__name__)
app.secret_key = 'your_secret_key'
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

@app.route('/', methods=['GET', 'POST'])
def index():
    result = session.get('result')
    print(result)
    return render_template('index.html', result=result)
def process_image(img_path):
    def open_img(img_path):
        carplate_img = cv2.imread(img_path)
        carplate_img = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)
        return carplate_img

    def carplate_extract(image, carplate_haar_cascade):
        carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in carplate_rects:
            carplate_img = image[y+15:y+h-10, x+15:x+w-20]

        return carplate_img

    def enlarge_img(image, scale_percent):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        return resized_image

    carplate_img_rgb = open_img(img_path)
    carplate_haar_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    carplate_extract_img = carplate_extract(carplate_img_rgb, carplate_haar_cascade)
    carplate_extract_img = enlarge_img(carplate_extract_img, 150)

    carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)
    carplate_text = pytesseract.image_to_string(
        carplate_extract_img_gray,
        config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    return carplate_extract_img_gray, carplate_text

@app.route('/save_photo', methods=['POST'])
def save_photo():
    image_data = request.json.get('image')

    # Запускаем скрипт для обработки
    processed_img, carplate_text = process_image(img_path='5.jpeg')

    result = f'Номер авто: {carplate_text}'
    print(result)

    session['result'] = result

    return redirect(url_for('index'))




if __name__ == '__main__':
    app.run()