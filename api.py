from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def post():
    if rfc:
        try:
            json_ = request.json
            query = (pd.DataFrame(json_))
            print(query)
            query = encoder.transform(query)

            print(query)
            prediction = list(rfc.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    rfc = joblib.load('rfcModel.pkl')
    encoder = joblib.load('encoder.joblib')
    print('model loaded')
    app.run(debug=True)
