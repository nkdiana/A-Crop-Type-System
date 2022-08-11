import numpy as np 
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('GradientBoost.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():   
     N = int(request.form['nitrogen'])
     P = int(request.form['phosphorus'])
     K = int(request.form['potassium'])
     Temperature = float(request.form['temperature'])
     Humidity = float(request.form['humidity'])
     ph = float(request.form['ph'])
     rainfall = float(request.form['rainfall'])
    
     final_features = np.array([[N,P,K,Temperature,Humidity,ph,rainfall]])
     prediction = model.predict(final_features)
    
     output = prediction[0]
    
     return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run()