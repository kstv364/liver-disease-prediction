import numpy as np
from flask import Flask, request, jsonify, render_template
from model import MLP
import pickle

app = Flask(__name__)
model = pickle.load(open('model64.pkl', 'rb'))
stats = pickle.load(open('stats.pkl', 'rb'))
print(len(stats['mean']))
print(len(stats['std']))
stats['mean'].insert(1,0)
stats['std'].insert(1,1)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # print(request.form.values())
    
    mean = np.array(stats['mean'])
    std = np.array(stats['std']) 
    features = [float(x) for x in request.form.values()]
    final_features = (np.array(features)-mean)/std
    prediction = model.predict(np.array(final_features))
    print("output",round(prediction[0][0],10))
    output = int(round(prediction[0][0]))
    if(output > 0):
        prediction = 'Liver disease is likely with 75% probability'
    else:
        prediction = 'Liver disease is unlikely'

    return render_template('index.html', prediction_text=prediction)


if __name__ == "__main__":
    app.run(debug=True)