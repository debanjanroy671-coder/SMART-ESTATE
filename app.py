from flask import Flask, render_template, send_from_directory, request, jsonify
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__, template_folder=".")


# =====================================================
# LOAD DATASET
# =====================================================

data = pd.read_csv("flats.csv")


# =====================================================
# FEATURES & TARGET
# =====================================================

X = data[
    [
        "Area_sqft",
        "Facing",
        "Floor",
        "Parking_sqft",
        "Bedrooms"
    ]
]

y = data["Price_Lakhs"]


# =====================================================
# PREPROCESSING
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "facing",
            OneHotEncoder(handle_unknown="ignore"),
            ["Facing"]
        )
    ],
    remainder="passthrough"
)


# =====================================================
# AI MODEL
# =====================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


# =====================================================
# TRAIN MODEL
# =====================================================

model.fit(X, y)


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():
    return render_template("index.html")


# =====================================================
# CSS
# =====================================================

@app.route("/style.css")
def style():
    return send_from_directory(".", "style.css")


# =====================================================
# PREDICT API
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data_received = request.get_json()

        area = float(data_received["Area_sqft"])
        facing = data_received["Facing"]
        floor = int(data_received["Floor"])
        parking = float(data_received["Parking_sqft"])
        bedrooms = int(data_received["Bedrooms"])


        input_data = pd.DataFrame(
            [
                {
                    "Area_sqft": area,
                    "Facing": facing,
                    "Floor": floor,
                    "Parking_sqft": parking,
                    "Bedrooms": bedrooms
                }
            ]
        )


        prediction = model.predict(input_data)[0]


        return jsonify(
            {
                "success": True,
                "price": round(float(prediction), 2)
            }
        )


    except Exception as error:

        return jsonify(
            {
                "success": False,
                "error": str(error)
            }
        ), 400


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)
