# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:28:44 2019

@author: djeevana
"""
from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df= pd.read_excel(r'C:\Projects\config_classification\Problem_Report_Full.xlsx',usecols=['Problem RCA Analysis','RCA Code'])
    df.dropna(how='any',inplace=True)
    df['is_config'] = np.where(df['RCA Code'].str.contains('Configuration/Set Up'), 1, 0)
    X = df['Problem RCA Analysis']
    y = df['is_config']
	
    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X) # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

    if request.method == 'POST':
        problem_rca = request.form['Problem RCA Analysis']
        data = [problem_rca]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	#app.run(debug=True)
    #app.debug == True
    
    #app.run(port=8001, debug=True)
    
    p4ort = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)

