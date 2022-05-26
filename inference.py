import numpy
import datetime
import pandas
# ===========================================================
from Helper import Funcs
from matplotlib import pyplot as plt
from keras.models import load_model
# ===========================================================
hp = Funcs()  # Init Hepler class
# ===========================================================
Data = hp.Get_Historicalprice()  # get data
Data = hp.Normalize(Data)  # normalize data
# ===========================================================
model = load_model('model.h5')
print('\n model.h5 loaded successfully\n')
# ===========================================================
daystopredict = 180
window_size = 7
features = 1
nextdays = hp.build_NextDays(Data, daystopredict, window_size, model)
predictions = hp.Denormalize(nextdays.values)
# ===========================================================
for val in range(len(nextdays)):
    print(str(nextdays.index[val]).split(' ')[0]+' ------> {}'.format(predictions[val][0]))
# ===========================================================]
#Plot predict data
plt.figure(figsize=(10,8))
plt.plot(Data[365:])
plt.plot(nextdays, color= 'red', label = 'prediction' )
plt.xlabel('Time in months', size = 16)
plt.ylabel('Decrease -<->+ Increase', size = 16)
plt.title('Bitcoin price prediction', color='g', size = 20, pad = 20)
plt.legend()
plt.show()
# ===========================================================
#End 1
#Saving Pred data to text

temp_predata = []   #temporary storage of predicted data
f_date = datetime.datetime.now() #file create date
temp_predata.append(f_date.strftime("File creation Date : %Y-%m-%d"'\nTime :%H:%M:%S.%f\n'))


for val in range(len(nextdays)):
    temp_predata.append((str(nextdays.index[val]).split(' ')[0]+' ------> {}'.format(predictions[val][0])))

with open('Prediction_data.txt', 'w') as file:
    for item in temp_predata:
        file.write("%s\n" % item)
print('\n Prediction data saved to file \n')
# ===
print("\n END OF EXECUTION \n")

# ===========================================================
#End End