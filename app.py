# save this as app.py
from flask import Flask, escape, request, render_template, jsonify
from flask_cors import cross_origin
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import keras.utils as image
# from PIL import Image

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# model = load model("models/fruits.h5")
model = load_model("models/1")

class_name = ['Apples',
              'Banana',
              'Oranges',
              'Apples',
              'Banana',
              'Oranges']

app = Flask(__name__)


@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return jsonify(result='welcome...')  # render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/api/v1/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        f = request.files['fruit'].read()
        filename = request.form.get('filename')
        target = os.path.join(APP_ROOT, 'images/')
        # print(target)
        des = "/".join([target, filename])
        # f.save(des)

        with open(des, 'wb') as file:
            file.write(f)

        # test_image = Image.open('images\\'+filename)
        test_image = image.load_img(
            'images//'+filename, target_size=(300, 300))
        # print(test_image)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        prediction = model.predict(test_image)
        # print(prediction)

        predicted_class = class_name[np.argmax(prediction[0])]
        # print(predicted_class)
        confidence = round(np.max(prediction[0])*100)
        # print(confidence)

        return jsonify(confidence=confidence,
                       prediction=prediction)

    else:
        return jsonify(result='coudnt fetch api results', success=False)


if __name__ == '__main__':
    app.debug = True
    app.run()
