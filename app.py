import flask
from flask import render_template, request
from flask import jsonify
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import requests, json
from flask_mysqldb import MySQL
import MySQLdb.cursors
from flask import render_template, request, jsonify, session
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline 
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 
from sklearn.svm import LinearSVC

app = flask.Flask(__name__, template_folder='Templates')

#code for connection
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'elect'


#Read Data
ds1 = pd.read_csv('Data/bjp_tweets.csv', nrows=10000)
ds2 = pd.read_csv('Data/congress_tweets.csv',  nrows=10000)
data = pd.concat([ds1, ds2], ignore_index=True)

def removeHTML(raw_text):
    clean_HTML = BeautifulSoup(raw_text, 'lxml').get_text() 
    return clean_HTML
def removeSpecialChar(raw_text):
    clean_SpecialChar = re.sub("[^a-zA-Z]", " ", raw_text)  
    return clean_SpecialChar
def toLowerCase(raw_text):
    clean_LowerCase = raw_text.lower().split()
    return( " ".join(clean_LowerCase)) 
def removeStopWords(raw_text):
    stops = set(stopwords.words("english"))
    words = [w for w in raw_text if not w in stops]
    return( " ".join(words))

tvec = TfidfVectorizer(use_idf=True,
strip_accents='ascii')

login = 0
#Train and Test Data Split

X = data['tweet']
Y = data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)

# X_training clean set
X_train_cleaned = []

for val in X_train:
    val = removeHTML(val)
    val = removeSpecialChar(val)
    val = toLowerCase(val)
    X_train_cleaned.append(val) 
    
# X_testing clean set
X_test_cleaned = []

for val in X_test:
    val = removeHTML(val)
    val = removeSpecialChar(val)
    val = toLowerCase(val)
    X_test_cleaned.append(val) 
    
X_train_tvec = tvec.fit_transform(X_train_cleaned)

mysql = MySQL(app)
@app.route('/')

@app.route('/main', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':

        return(flask.render_template('index.html'))
       
            
    
@app.route('/about', methods=['GET', 'POST'])
def about():
    if flask.request.method == 'GET':
        return(flask.render_template('about.html'))
    
@app.route('/services', methods=['GET', 'POST'])
def services():
    if flask.request.method == 'GET':
        return(flask.render_template('services.html'))

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if flask.request.method == 'GET':
        return(flask.render_template('contact.html'))

@app.route('/mainpage', methods=['GET', 'POST'])
def mainpage():
    if flask.request.method == 'GET':
        return(flask.render_template('mainpage.html'))

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    if flask.request.method == 'GET':
        login = 0
        return(flask.render_template('index.html'))

model = pickle.load(open('Model/election_analyser.pkl', 'rb'))

import matplotlib.pyplot as plt
import random
@app.route('/getprediction', methods=['GET', 'POST'])
def getprediction():
    if flask.request.method == 'POST':
        search           = request.form['search']
        limit           = request.form['limit']
        url = f"https://tweetcracks.000webhostapp.com/?search={search}&limit={limit}"

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for tweet in data:
                cont = tweet["tweet_content"]
                
                cont = removeHTML(cont)
                cont = removeSpecialChar(cont)
                cont = toLowerCase(cont)

                sentimentdt = [str(cont)]
                prediction = model.predict(tvec.transform(sentimentdt))[0]
                
                if prediction == 0:
                    pred = "Negative"
                else:
                    pred = 'Positive'
                    
                tweet["prediction"] = pred
                
            df = pd.DataFrame(data)
            party_counts = df.groupby(['party', 'prediction']).size().unstack(fill_value=0)
            
            

            # Create comparison graph with custom colors and labels
            colors = ['#1f77b4', '#ff7f0e']  # Blue for Positive, Orange for Negative
            party_counts.plot(kind='bar', stacked=True, color=colors)
            plt.xlabel('Party')
            plt.ylabel('Count')
            plt.title('Comparison of Positive and Negative Reviews by Party')
            plt.xticks(rotation=45)
            plt.legend(title='Prediction')
            
            
            
            # Filter unique parties
            unique_parties = df['party'].unique()
            
            # Dictionary to store positive and negative counts for each party
            party_counts = {}
            
            # Iterate over unique parties
            for party in unique_parties:
                # Filter data for the current party
                party_data = df[df['party'] == party]
                
                # Count positive and negative predictions
                positive_count = party_data[party_data['prediction'] == 'Positive'].shape[0]
                negative_count = party_data[party_data['prediction'] == 'Negative'].shape[0]
                
                # Store counts in dictionary
                party_counts[party] = {'Positive': positive_count, 'Negative': negative_count}
            
            # Determine the winner
            winner = max(party_counts.items(), key=lambda x: x[1]['Positive'])
            
            # Print party counts and winner
            allanal = ''
            for party, counts in party_counts.items():
                allanal += f"<b> {party} </b> : Positive - {counts['Positive']}, Negative - {counts['Negative']}<br>"
            winner = f"<b>{winner[0]} </b>  more positive tweets. Hence, The winning probability is high for {winner[0]} "
            
            out = {"tweet":data, "winner":winner, "analysis":allanal}
            # Save chart image in static folder
            plt.savefig('static/comparison_chart.png')

            return jsonify(out)
        else:
            return jsonify({"error": "Failed to fetch data"}), 500
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if flask.request.method == 'POST':
        phone           = request.form['signphone']
        password        = request.form['signpassword']
        
        con = mysql.connect
        con.autocommit(True)
        cursor = con.cursor(MySQLdb.cursors.DictCursor)
        qry = 'SELECT * FROM userdetail WHERE phone="'+phone+'" AND password="'+password+'"'
        result = cursor.execute(qry)
        result = cursor.fetchone()
        if result:
            login = 1
            msg = "1"
        else:
           msg = "0"
    return msg


@app.route('/register', methods=['GET', 'POST'])
def register():
    if flask.request.method == 'POST':
        username  = request.form['regusername']
        phone        = request.form['regphone']
        email        = request.form['regemail']
        password     = request.form['regpassword']
        
        con = mysql.connect
        con.autocommit(True)
        cursor = con.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('INSERT INTO userdetail VALUES (NULL, % s, % s, % s, %s, NULL)', (username, phone, email,  password, ))
        mysql.connect.commit()
        msg = '1'
        
        return msg

if __name__ == '__main__':
    app.run(debug=True)