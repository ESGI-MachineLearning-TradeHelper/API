from flask import Flask, request, Response
from PIL import Image
import jsonpickle
import tensorflow as tf
import numpy as np
import base64

target_size = (128, 128)
small_size = (64, 64)

app = Flask("DEEPLEARNING")
app.config["DEBUG"] = True

@app.route('/api/predict', methods=['POST'])
def predict():
    r = request
    type = int(r.args.get("type"))
    if not r.form:
        return Response(status=400)
    form = r.form.to_dict()
    if not form['image[content]']:
        return Response(status=400)
    else:
        print(form['image[content]'])
        stripped_base64image = form['image[content]'].split(',')[1]
        print(stripped_base64image)
        img_data = base64.b64decode(stripped_base64image)
        filename = form['image[name]']
        with open("img/" + filename, 'wb') as f:
            f.write(img_data)
        # CALL FOR MODEL
        if type == 1:
            x_train = load_dataset(filename, 'L', target_size)
            print(x_train.shape)
            model = tf.keras.models.load_model('./models/linear.keras')
            resp = model.predict(x_train)
        elif type == 2:
            x_train = load_dataset(filename, 'RGB', target_size)
            print(x_train.shape)
            model = tf.keras.models.load_model('./models/cnn.keras')
            resp = model.predict(x_train)
        elif type == 3:
            x_train = load_dataset(filename, 'L', target_size)
            print(x_train.shape)
            model = tf.keras.models.load_model('./models/perceptron.keras')
            resp = model.predict(x_train)
        elif type == 4:
            x_train = load_dataset(filename, 'RGB', small_size)
            print(x_train.shape)
            model = tf.keras.models.load_model('./models/rnn.keras')
            resp = model.predict(x_train)
        else:
            x_train = load_dataset(filename, 'RGB', target_size)
            print(x_train.shape)
            model = tf.keras.models.load_model('./models/unet.keras')
            resp = model.predict(x_train)

        # RETURN OF MODE
        if resp[0][0] > resp[0][1] and resp[0][0] > resp[0][2]:
            response_pickled = jsonpickle.encode({'predict': "Rectangle", 'predict_percent': str(resp[0][0])})
        elif resp[0][1] > resp[0][0] and resp[0][1] > resp[0][2]:
            response_pickled = jsonpickle.encode({'predict': "Flag", 'predict_percent': str(resp[0][1])})
        elif resp[0][2] > resp[0][0] and resp[0][2] > resp[0][1]:
            response_pickled = jsonpickle.encode({'predict': "Double bottom", 'predict_percent': str(resp[0][2])})
        else:
            response_pickled = jsonpickle.encode({'predict':"None", 'predict_percent':0})

        print(resp)
        return Response(response=response_pickled, status=200, mimetype="application/json")

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

def load_dataset(filename, type, size):
    test_path = "img/"
    x_test_list = []
    load_set_from_directory(test_path, x_test_list, filename, type, size)
    return np.array(x_test_list)

def load_set_from_directory(train_path, x_train_list, filename, type, size):
    load_image_from_directory(train_path, x_train_list, filename, type, size)

def load_image_from_directory(path, x_train_list, filename, type, size):
    x_train_list.append(
            np.array(Image.open(path + filename).convert(type).resize(size)) / 255.0)  # color


app.run(host="0.0.0.0", port=5000)
