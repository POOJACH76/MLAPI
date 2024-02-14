from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("MODELS/Stress_Model.pkl", "rb"))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        p1 = float(data['Humidity'])
        p2 = float(data['Temperature'])
        p3 = int(data['Step_count'])

        predictions = model.predict(np.array([[p1, p2, p3]]))
        probability = model.predict_proba(np.array([[p1, p2, p3]]))[0, 1] * 100

        return jsonify({'Stress prediction': predictions.tolist(), 'Probability': probability})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7050, debug=True)
