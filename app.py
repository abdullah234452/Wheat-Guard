from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)


# Load trained model
with open("health_model (1).pkl", "rb") as file:
    model_data = pickle.load(file)


model = model_data["model"]
crop_encoder = model_data["crop_encoder"]
soil_encoder = model_data["soil_encoder"]
stage_encoder = model_data["stage_encoder"]
result_encoder = model_data["result_encoder"]


# Home page
@app.route("/")
def home():

    crops = crop_encoder.classes_.tolist()
    soils = soil_encoder.classes_.tolist()
    stages = stage_encoder.classes_.tolist()

    return render_template(
        "index.html",
        crops=crops,
        soils=soils,
        stages=stages
    )


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        crop = data["crop"]
        soil = data["soil"]
        stage = data["stage"]

        moi = float(data["moi"])
        temp = float(data["temp"])
        humidity = float(data["humidity"])


        # Convert text into numbers
        crop_encoded = crop_encoder.transform([crop])[0]
        soil_encoded = soil_encoder.transform([soil])[0]
        stage_encoded = stage_encoder.transform([stage])[0]


        # Create input
        new_data = pd.DataFrame([
            [
                crop_encoded,
                soil_encoded,
                stage_encoded,
                moi,
                temp,
                humidity
            ]
        ], columns=[
            "crop ID",
            "soil_type",
            "Seedling Stage",
            "MOI",
            "temp",
            "humidity"
        ])


        # Prediction
        prediction = model.predict(new_data)

        # Convert prediction back to text
        result = result_encoder.inverse_transform(prediction)


        return jsonify({
            "success": True,
            "crop_health": result[0]
        })


    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run()