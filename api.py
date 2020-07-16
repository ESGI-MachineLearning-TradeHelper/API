from flask import Flask, request, Response
from PIL import Image
import jsonpickle
import tensorflow as tf
import numpy as np
import base64

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
        stripped_base64image = form['image[content]'].split(',')[1]
        img_data = base64.b64decode(stripped_base64image)
        filename = "img/" + form['image[name]']
        with open(filename, 'wb') as f:
            f.write(img_data)
        # CALL FOR MODEL
        if type == 1:
            model = tf.keras.models.load_model('models/linear.keras')
            imgFrom = Image.open("img/" + filename)
            imgResized = imgFrom.convert("L").resize((128, 128))
            npArray = np.array(imgResized) / 255
            resp = model.predict(npArray[None, :])
        elif type == 2:
            model = tf.keras.models.load_model('models/cnn.keras')
            imgFrom = Image.open("img/" + filename)
            imgResized = imgFrom.convert("RGB").resize((128, 128))
            npArray = np.array(imgResized) / 255
            resp = model.predict(npArray[None, :])
        elif type == 3:
            model = tf.keras.models.load_model('models/perceptron.keras')
            imgFrom = Image.open("img/" + filename)
            imgResized = imgFrom.convert("L").resize((128, 128))
            npArray = np.array(imgResized) / 255
            resp = model.predict(npArray[None, :])
        elif type == 4:
            model = tf.keras.models.load_model('models/rnn.keras')
            imgFrom = Image.open("img/" + filename)
            imgResized = imgFrom.convert("RGB").resize((64, 64))
            npArray = np.array(imgResized)
            npExpand = np.expand_dims(npArray, axis=0)
            resp = model.predict(npExpand)
        else:
            model = tf.keras.models.load_model('models/unet.keras')
            imgFrom = Image.open("img/" + filename)
            imgResized = imgFrom.convert("RGB").resize((128, 128))
            npArray = np.array(imgResized)
            npExpand = np.expand_dims(npArray, axis=0)
            resp = model.predict(npExpand)
        # RETURN OF MODE
        if resp[0][0] > resp[0][1] and resp[0][0] > resp[0][2]:
            response_pickled = jsonpickle.encode({'predict': "Rectange", 'predict_percent': resp[0][0]})
        if resp[0][1] > resp[0][0] and resp[0][1] > resp[0][2]:
            response_pickled = jsonpickle.encode({'predict': "Flag", 'predict_percent': resp[0][1]})
        if resp[0][2] > resp[0][0] and resp[0][2] > resp[0][1]:
            response_pickled = jsonpickle.encode({'predict': "Double bottom", 'predict_percent': resp[0][2]})
        else:
            response_pickled = jsonpickle.encode({'predict':"None", 'predict_percent':0})

        return Response(response=response_pickled, status=200, mimetype="application/json")


app.run(host="0.0.0.0", port=5000)
