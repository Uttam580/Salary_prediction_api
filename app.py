import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])#post method ,give feature to model.pkl file ,model take input and give output.
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]# take input from all the form  and store into feature 
    final_features = [np.array(int_features)]# converting into array
    prediction = model.predict(final_features)# make prediction

    output = round(prediction[0], 2)# retrieve output

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))# agin rendering index.html and will data


if __name__ == "__main__":
    app.run(debug=True)# main function to render whole flask
