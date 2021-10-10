# import framework libraries 
from django.shortcuts import render
import streamlit as st
import streamlit.components.v1 as compo 
from datetime import date

#Import ML libraries 
import yfinance as yf
from sklearn.linear_model import  LinearRegression as lr #
from sklearn.preprocessing import MinMaxScaler as mms
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from plotly import graph_objs as go
import numpy as np
import tensorflow as tf

#Importing sentimental analysis/scrapping libraries
import snscrape.modules.twitter as snstweet
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sn
import itertools as it 
import collections as cs

import tweepy as tw
import nltk
from nltk.corpus import stopwords
import re
import networkx as nx 
from textblob import TextBlob
import warnings 

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Predicting App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME','BOA',)
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data , render('THome.html')
	
	


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	return render('THome.html'), st.plotly_chart(fig)
	
   


plot_raw_data()

# Predict forecast with Keras.

#Training and testing data set
dataTrain = data[['Date','Close']]
dataTrain = dataTrain.rename(columns={"Date": "ds", "Close": "y"})
nData = dataTrain.loc[884:1639]
nData = nData.drop('Date', axis= 1)
ndData = nData.reset_index(drop=True)

#Scaler
scaler = mms(feature_range=(0,1))
T = scaler.fit_transform

T = T.astype('float32')
T= np.reshape(T,(-1,1))


#Scaler
scaler = mms(feature_range=(0,1))
T = scaler.fit_transform(T)

 
trainSize = int(len(T)*0.88)
testSize = int(len(T)-trainSize)
train, test = T[0:trainSize, :], T[trainSize:len(T):, ]

def createFeatures(data,windowSize):
	X,Y = [], []
	for i in range(len(data)-windowSize - 1):
		window = data[i:(i + windowSize), 0]
	X.append(window)
	Y.append(window)
	return (np.array(X), np.array(Y))

windowSize = 20

X_Train, Y_Train = createFeatures(train, windowSize)

X_test, Y_test = createFeatures(test, windowSize)

# Reshape to the format of [samples, time steps, features]
X_train = np.reshape(X_Train, (X_Train.shape[0], 1, X_Train.shape[1]))

X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

T_shape = T.shape
train_shape = train.shape
test_shape = test.shape

# Make sure that the number of rows in the dataset = train rows + test rows
def isLeak(T_shape, train_shape, test_shape):
    return not(T_shape[0] == (train_shape[0] + test_shape[0]))

tf.random.set_seed(11)
np.random.seed(11)

# Building model
model = Sequential()

model.add(LSTM(units = 50, activation = 'relu', #return_sequences = True, 
               input_shape = (X_train.shape[1], windowSize)))
model.add(Dropout(0.2))

# Optional additional model layer to make a deep network. If you want to use this, uncomment #return_sequences param in previous add
"""
model.add(LSTM(units = 25, activation = 'relu'))
model.add(Dropout(0.2))
"""

# Output layer
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# Save models
filepath = 'saved_models/model_epoch_{epoch:02d}.hdf5'

checkpoint = ModelCheckpoint(filepath = filepath,
                             monitor = 'val_loss',
                             verbose = 1,
                             save_best_only = True,
                             mode ='min'
                            )

history = model.fit(X_train, Y_Train, epochs = 100, batch_size = 20, validation_data = (X_test, Y_test), 
                    callbacks = [checkpoint], 
                    verbose = 1, shuffle = False)


best_model = load_model('saved_models/model_epoch_89.hdf5')

# Predicting and inverse transforming the predictions

train_predict = best_model.predict(X_train)

Y_hat_train = scaler.inverse_transform(train_predict)

test_predict = best_model.predict(X_test)

Y_hat_test = scaler.inverse_transform(test_predict)

# Inverse transforming the actual values, to return them to their original values
Y_test = scaler.inverse_transform([Y_test])
Y_train = scaler.inverse_transform([Y_Train])

# Reshaping 
Y_hat_train = np.reshape(Y_hat_train, newshape = 583)
Y_hat_test = np.reshape(Y_hat_test, newshape = 131)

Y_train = np.reshape(Y_train, newshape = 583)
Y_test = np.reshape(Y_test, newshape = 131)

from sklearn.metrics import mean_squared_error

train_RMSE = np.sqrt(mean_squared_error(Y_train, Y_hat_train))

test_RMSE = np.sqrt(mean_squared_error(Y_test, Y_hat_test))


def get_or_not():
 os.system("snscrape --jsonl --max-results 70 --since 2020-06-01 twitter-search \"AAPL until:2021-8-19\" > text-query-tweets.json")
tweets = pd.read_json("text-query-tweets.json", lines = True)
tweetlist1 = []
for i,tweet in enumerate(snstweet.TwitterSearchScraper('AAPL').get_items()):
	if i>70:
		break  
	tweets_df1 = pd.DataFrame(tweetlist1, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
	




	
	






