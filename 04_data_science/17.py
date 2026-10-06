import os
import pickle

from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the pre-trained model once, when the server starts
pickle_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_estate_model.pkl')
with open(pickle_path, 'rb') as f:
    model = pickle.load(f)

# What the model was trained on: any other input must be rejected
ALLOWED_LOCATIONS = ('Downtown', 'Suburb', 'Rural')
ALLOWED_AMENITIES = ('Basic', 'Standard', 'Premium')
NUMERIC_RANGES = {
    'size': (200, 10000),  # square feet
    'year_built': (1800, 2030),
    'num_bedrooms': (0, 20),
    'num_bathrooms': (0, 20),
}


def validate(data):
    """Returns a list of problems with the request. An empty list means the request is valid."""
    if not isinstance(data, dict):
        return ['Request body must be a JSON object']
    errors = []
    if data.get('location') not in ALLOWED_LOCATIONS:
        errors.append(f"location must be one of {list(ALLOWED_LOCATIONS)}")
    if data.get('amenities') not in ALLOWED_AMENITIES:
        errors.append(f"amenities must be one of {list(ALLOWED_AMENITIES)}")
    for field, (low, high) in NUMERIC_RANGES.items():
        value = data.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            errors.append(f"{field} must be a number between {low} and {high}")
    return errors


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    # Extract and validate the features from the POST request
    data = request.get_json(silent=True)
    errors = validate(data)
    if errors:
        return jsonify({'errors': errors}), 400

    # The model expects the six features in exactly this order
    features = [data['location'], data['size'], data['amenities'],
                data['year_built'], data['num_bedrooms'], data['num_bathrooms']]

    # Make prediction
    prediction = model.predict([features])

    # Return the prediction as a JSON object
    return jsonify({'predicted_price': round(float(prediction[0]), 2)})


if __name__ == '__main__':
    # Development server only. In production, run the app behind a WSGI server such as gunicorn.
    app.run(port=5000)
