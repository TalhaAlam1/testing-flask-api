import numpy as np
from flask import Blueprint, request, jsonify
import pickle
import os

main = Blueprint('main', __name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'D:/vercel deployment/finalized_model.sav')
model = pickle.load(open(model_path, 'rb'))

@main.route('/api', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            if not data:
                return jsonify({"error": "Empty request"}), 400
            features = [np.array(list(data.values()))]
            prediction = model.predict(features)
            output = float(prediction[0])  # Convert np.float32 to standard float
            return jsonify(output)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    else:  # GET request
        return "This is the API endpoint. Please use POST method to get predictions."
