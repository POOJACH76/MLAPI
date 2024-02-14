from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load label encoders
with open("MODELS/Hair_encoding.pkl", "rb") as file:
    label_encoders = pickle.load(file)

# Load the ML model
with open("MODELS/Hair_Loss.pkl", "rb") as file:
    model = pickle


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Prepare input data for prediction
        input_data = np.array([

            data['stay_up_late'],
            data['pressure_level'],
            data['coffee_consumed'],
            data['brain_working_duration'],
            data['shampoo_brand'],
            data['swimming'],
            data['hair_washing'],
            data['dandruff']
        ]).reshape(1, -1)

        # Make predictions
        predictions = model.predict(input_data)
        probability = model.predict_proba(input_data)

        # Assuming you want the probability of the first class
        probability_first_class = probability[0, 1, 2, 3] * 100

        return jsonify({'MentalHealth prediction': int(predictions[0]), 'Probability': probability_first_class})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7090, debug=True)
