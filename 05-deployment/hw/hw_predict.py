import pickle

from flask import Flask, request, jsonify

# Load the DictVectorizer
with open('dv.bin', 'rb') as dv_file:
    dv = pickle.load(dv_file)

# Load the model
with open('model2.bin', 'rb') as model_file:
    model = pickle.load(model_file)

# Now you can use the dict_vectorizer and model as needed
client = {"job": "management", "duration": 400, "poutcome": "success"}


def predict_one(data):
    X = dv.transform([data])
    y_pred = model.predict_proba(X)[0, 1]
    return y_pred

def predicted_fixed_client():
    prediction = predict_one(client)
    # What's the probability that this client will get a subscription?
    client_subscription_proba = float(prediction).__round__(3)
    print(f"{client=}")
    print(f"{client_subscription_proba=}")


app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    y_pred = predict_one(customer)
    churn = y_pred >= 0.5

    result = {
        'subscription_probability': float(y_pred),
        'subscription': bool(churn),
    }

    return jsonify(result), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
