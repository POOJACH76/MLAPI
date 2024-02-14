from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("MODELS/PCOS.pkl", "rb"))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        p1 = int(data['Age'])
        p2 = float(data['Weight (Kg)'])
        p3 = float(data['Height_Cm'])
        p4 = float(data['BMI'])
        p5 = str(data['Blood Group'])
        p6 = int(data['Pulse_rate_bpm'])
        p7 = int(data['RR_breaths_min'])
        p8 = int(data['Cycle_length_days'])
        p9 = float(data['Marriage_Status_Yrs'])
        p10 = int(data['Pregnant'])
        p11 = int(data['Hip_inch'])
        p12 = int(data['Waist_inch'])
        p13 = float(data['Waist_Hip_Ratio'])
        p14 = int(data['Weight_gain'])
        p15 = int(data['Hair_growth'])
        p16 = int(data['Skin_darkening'])
        p17 = int(data['Hair_loss'])
        p18 = int(data['Pimples'])
        p19 = float(data['Fast_food'])
        p20 = int(data['Reg_Exercise'])
        p21 = int(data['BP_Systolic_mmHg'])
        p22 = int(data['BP_Diastolic_mmHg'])

        prediction = model.predict(np.array(
            [[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22]]))
        predictions_list = prediction.tolist()
        probability = model.predict_proba(np.array(
            [[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22]]))[
                          0, 1] * 100
        return jsonify({'Cardio prediction': predictions_list, 'Probability': probability})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7030, debug=True)
