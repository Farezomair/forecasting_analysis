import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model_path = r"C:\Users\Farez Laptop\Dropbox\My PC (DESKTOP-0DN7O4V)\Desktop\Capstone Project\regression_model.joblib"
model = joblib.load(model_path)

# Initialize Flask app
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
template_dir = os.path.join(current_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)

@app.route("/")
def home():
    return render_template("index.html")  # Ensure index.html exists

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features_list = data.get("features", [])

        # Validate input
        if not isinstance(features_list, list) or len(features_list) != 9:
            return jsonify({"error": "Please enter exactly 9 numerical features!"}), 400

        # Convert input to NumPy array and reshape for model
        features_array = np.array(features_list).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)

        return jsonify({"predicted_stock_price": float(prediction[0])})  # Ensure output is JSON serializable
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
