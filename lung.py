from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("MODELS/Lung_Cancer.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        p1 = int(data['AGE'])
        p2 = int(data['SMOKING'])
        p3 = float(data['YELLOW_FINGERS'])
        p4 = int(data['ANXIETY'])
        p5 = int(data['PEER_PRESSURE'])
        p6 = float(data['CHRONIC DISEASE'])
        p7 = int(data['FATIGUE'])
        p8 = int(data['ALLERGY'])
        p9 = int(data['WHEEZING'])
        p10 = float(data['ALCOHOL CONSUMING'])
        p11 = int(data['COUGHING'])
        p12 = int(data['SHORTNESS OF BREATH'])
        p13 = int(data['SWALLOWING DIFFICULTY'])
        p14 = int(data['CHEST PAIN'])

        predictions = model.predict(np.array([[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]]))

        predictions_list = predictions.tolist()
        probability = \
            model.predict_proba(np.array([[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, ]]))[1, 2] * 100
        return jsonify({'Cardio prediction': predictions_list, 'Probability': probability})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7060, debug=True)
