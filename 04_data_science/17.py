from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load the pre-trained model
pickle_path = os.path.join(os.path.dirname(__file__), 'real_estate_model.pkl')
model = pickle.load(open(pickle_path, 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the POST request
    data = request.get_json(force=True)
    features = [data['location'], data['size'],data['amenities'], data['year_built'], data['num_bedrooms'], data['num_bathrooms']]

    # Make prediction
    prediction = model.predict([features])

    # Return the prediction as a JSON object
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)