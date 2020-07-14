from flask import Flask, request, Response
from PIL import Image
import jsonpickle

app = Flask("DEEPLEARNING")
app.config["DEBUG"] = True


@app.route('/api/predict', methods=['POST'])
def predict():
    r = request
    type = r.args.get("type")
    if not r.files:
        return Response(status=400)
    elif not r.files['image']:
        return Response(status=400)
    else:
        file = r.files['image']
        img = Image.open(file.stream)
        img = img.save("img/" + file.filename)
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
        response_pickled = jsonpickle.encode({'msg': 'success'})
        return Response(response=response_pickled, status=200, mimetype="application/json")


app.run(host="0.0.0.0", port=5000)
