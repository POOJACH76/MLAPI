from flask import Flask, request, jsonify
import pickle
import numpy as np


app = Flask(__name__)
with open('MODELS/Mental_Health.pkl', "rb") as file:
    model = pickle


@app.route("/predict_mentalhealth", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        p1 = int(data['Age'])
        p2 = int(data['Gender'])
        p3 = float(data['self_employed'])
        p4 = int(data['family_history'])
        p6 = float(data['work_interfere'])
        p7 = int(data['tech_company'])
        p8 = int(data['seek_help'])
        p9 = int(data['leave'])
        p10 = float(data['coworkers'])
        p11 = int(data['supervisor'])
        p12 = int(data['obs_consequence'])

        predictions = model.predict(np.array([[p1, p2, p3, p4, p6, p7, p8, p9, p10, p11, p12]]))

        predictions_list = predictions.tolist()
        probability = model.predict_proba(np.array([[p1, p2, p3, p4, p6, p7, p8, p9, p10, p11, p12]]))[0, 1] * 100
        return jsonify({'Cardio prediction': predictions_list, 'Probability': probability})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8020, debug=True)

