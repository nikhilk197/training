import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import keras
from keras.callbacks import EarlyStopping
dataset = pd.read_csv("fd909.csv").values
# split into input (X) and output (y) variables
X = dataset[:,0:12]
y = dataset[:,12]
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=12, activation='relu'))
model.add(Dense(423, activation='relu'))
model.add(Dense(423, activation ='relu'))
model.add(Dense(423, activation='relu'))
model.add(Dense(1))
# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
# fit the keras model on the dataset
earlystopping_callbacks = EarlyStopping(monitor='val_acc',verbose=1,min_delta=50,patience=3,baseline=None)
model.fit(X, y, epochs=10000, batch_size=10, callbacks = [earlystopping_callbacks])
te= np.array([215,0,0,226,0,0,255,0,0,255,0,0])
print (model.predict(np.reshape(te, (1,12)), batch_size = 1))
model.save('11601.h5')