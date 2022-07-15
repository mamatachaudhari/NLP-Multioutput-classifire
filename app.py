import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('transfrom.pkl','rb'))

app1 = Flask(__name__)


@app1.route('/')
def home():
    return render_template('student.html')

@app1.route('/result',methods=['POST', 'GET'])
def result():
    
    result = request.form["BP Code"]
    result_pred = model.predict(cv.transform([result]))
    out = result_pred.transpose()
    return render_template('final.html',out=out)




if __name__ == '__main__':
	app1.run(debug=True)
        

    
   