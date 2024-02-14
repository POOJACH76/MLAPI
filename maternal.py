from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load label encoders
with open("MODELS/Maternal_Encoders.pkl", "rb") as file:
    label_encoders = pickle.load(file)

# Load the ML model
with open("MODELS/Maternal_Health.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Apply label encoding to categorical features
        for feature in ['hair_loss', 'pressure_level', 'shampoo_brand', 'dandruff', 'hair_washing', 'swimming']:
            data[feature] = label_encoders[feature].transform([data[feature]])

        # Prepare input data for prediction
        input_data = np.array([
            int(data['Age']),
            int(data['SystolicBP']),
            float(data['DiastolicBP']),
            int(data['BS']),
            int(data['BodyTemp']),
            float(data['HeartRate']),


        ]).reshape(1, -1)

        # Make predictions
        predictions = model.predict(input_data)
        probability = model.predict_proba(input_data)[0, 1, 2] * 100

        return jsonify({'MentalHealth prediction': int(predictions[0]), 'Probability': probability})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7080, debug=True)
