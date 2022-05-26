import numpy
import datetime
import pandas

from Helper import Funcs
from matplotlib import pyplot as plt
from keras.models import load_model

print(" ===========================================================")
print(" Welcome to Bitcoin Price Predicon ver 2.0")
print(" ===========================================================")

hp = Funcs()  # Init Hepler class
# ===========================================================
Data = hp.Get_Historicalprice()  # get data
print(" Data     Shape : " + str(Data.shape))
# ===========================================================
Org_data = Data.copy()  # Make copy of org data -_o
# ===========================================================
Data = hp.Normalize(Data)  # normalize data
# ===========================================================
LastdaysForTest = 180
Train, Test = hp.Split_Test_Train(Data, LastdaysForTest)  # splitdata
print(" Train    Shape : " + str(Train.shape))
print(" Test     Shape : " + str(Test.shape))
# ===========================================================
window_size = 7
features = 1
# ===========================================================
X_Train = hp.Convert_TS_To_SL(Train, window_size)
X_Test = hp.Convert_TS_To_SL(Test, window_size)
Y_Train = Train[window_size:].values
Y_Test = Test[window_size:].values
print(" X_Train  Shape : " + str(X_Train.shape))
print(" X_Test   Shape : " + str(X_Test.shape))
print(" Y_Train  Shape : " + str(Y_Train.shape))
print(" Y_Test   Shape : " + str(Y_Test.shape))
# ===========================================================
ep = 30
bs = 4
LSTM_Model = hp.build_LSTM(window_size, features)
LSTM_Model.summary()
# ===========================================================
X_Train = numpy.array(X_Train.values)
X_Train = X_Train.reshape(len(X_Train), window_size, features)
Y_Train = Y_Train.reshape(len(Y_Train), )
# ===========================================================
print('\n Please Wait, Training the Model...\n')
history = LSTM_Model.fit(X_Train, Y_Train, epochs=ep, batch_size=bs)  # Train
# ===========================================================
X_Test = numpy.array(X_Test.values)
X_Test = X_Test.reshape(len(X_Test), window_size, features)
Y_Test = Y_Test.reshape(len(Y_Test), )  # Reshape Y_Test for Evaluate
# ===========================================================
PY_Test = LSTM_Model.predict(X_Test)  # Predict Y_Test
# ===========================================================
history = LSTM_Model.evaluate(X_Test, Y_Test)  # Evaluate Model -_o
# ===========================================================
Y_Test = Y_Test.reshape(len(Y_Test), features)  # Reshape Y_Test back
# ===========================================================
LSTM_Model.save("model.h5") #saving the model
print('\n model saved to model.h5')
#============================================================
print('\nY_Test\n\n',  Y_Test)
print('\nPY_Test\n\n', PY_Test)
# ===
plt.figure(figsize=(10,10))
plt.plot(hp.Denormalize(Y_Test), color='cyan', label= 'Test Real Price')
plt.plot(hp.Denormalize(PY_Test), color='black', label= 'Test Predict Price')
plt.xlabel('Days', size = 16)
plt.ylabel('Price [USD]', size = 16)
plt.title('Y, PY Test',color = 'g', size = 20, pad = 20)
plt.legend()
plt.show()
# ===========================================================
#End End
