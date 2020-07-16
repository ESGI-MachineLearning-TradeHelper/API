from flask import Flask, request, Response
from PIL import Image
import jsonpickle
import base64

app = Flask("DEEPLEARNING")
app.config["DEBUG"] = True

@app.route('/api/predict', methods=['POST'])
def predict():
    r = request

    type = r.args.get("type")
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
            print("MODEL 1")
        elif type == 2:
            print("MODEL 2")
        elif type == 3:
            print("MODEL 3")
        elif type == 4:
            print("MODEL 4")
        else:
            print("MODEL 5")
        # RETURN OF MODEL
        response_pickled = jsonpickle.encode({'predict': 'triangle','predict_percent':30})
        return Response(response=response_pickled, status=200, mimetype="application/json")


app.run(host="0.0.0.0", port=5000)
