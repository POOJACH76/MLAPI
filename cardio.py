from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("MODELS/Cardio_Disease.pkl", "rb"))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        p1 = int(data['Age'])
        p2 = int(data['gender'])
        p3 = int(data['height'])
        p4 = int(data['weight'])
        p5 = int(data['ap_hi'])
        p6 = int(data['ap_lo'])
        p7 = int(data['cholesterol'])
        p8 = int(data['gluc'])
        p9 = int(data['smoke'])
        p10 = int(data['alco'])
        p11 = int(data['active'])

        predictions = model.predict(np.array([[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]]))
        probability = model.predict_proba(np.array([[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]]))[0, 1] * 100

        return jsonify({'Cardio prediction': predictions.tolist(), 'Probability': probability})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7040, debug=True)
