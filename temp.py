import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import *
import numpy as np
import pandas as pd
from keras.models import Sequential,load_model
from keras.layers import Dense,Activation,advanced_activations
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from statsmodels.tsa.stattools import acf
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
from keras.layers import Bidirectional
import xlsxwriter

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def import_train_data():
    global v_train
    global df_train
    csv_file_path = filedialog.askopenfilename(filetypes=(("CSV Files", "*.csv"), ("All", "*.*")))
    print(csv_file_path)
    v_train.set(csv_file_path)
    df_train = pd.read_csv(csv_file_path)
    df_train['Toplam (MWh)'] = df_train['Toplam (MWh)'].str.replace('.', '')
    df_train['Toplam (MWh)'] = df_train['Toplam (MWh)'].str.replace(',', '.')
    df_train['Toplam (MWh)'] = df_train['Toplam (MWh)'].astype(float)

    

def import_test_data():
    global v_test
    global df_test
    csv_file_path = filedialog.askopenfilename(filetypes=(("CSV Files", "*.csv"), ("All", "*.*")))
    print(csv_file_path)
    v_test.set(csv_file_path)
    df_test = pd.read_csv(csv_file_path)
    df_test['Toplam (MWh)'] = df_test['Toplam (MWh)'].str.replace('.', '')
    df_test['Toplam (MWh)'] = df_test['Toplam (MWh)'].str.replace(',', '.')
    df_test['Toplam (MWh)'] = df_test['Toplam (MWh)'].astype(float)



def read_csv_file():
    listAll.delete(0, END)
    for col in df_train.columns:
        listAll.insert(END, col)


def get_active_delete(self):
    self.insert(END, listAll.get(ACTIVE))


def eject_active_delete(self):
    self.delete(ACTIVE)

def get_Target():
 
    radSel = radVar_train.get()
    t_perc = int(train_percentage.get())
    targets = listTarget.get(0, END)
    targets = np.asarray(targets)
    trainY = df_train.get(targets)
    t_row = trainY.shape[0]
    if radSel == 1:
        train_count = (t_perc/100)*t_row
        train_count = int(train_count)
        trainY = trainY[(t_row-train_count):t_row]
        #train_percentage.set(trainX.shape[0])
        #trainX = np.asarray(trainX)
    elif radSel == 2:
        trainY = trainY[:t_perc]
        
    trainY = np.asarray(trainY)
    return trainY

def get_Predictor():

    radSel = radVar_train.get()
    t_perc = int(train_percentage.get())
    predictors = listPredictor.get(0, END)
    predictors = np.asarray(predictors)
    trainX = df_train.get(predictors)
    t_row = trainX.shape[0]
    if radSel == 1:
        train_count = (t_perc/100)*t_row
        train_count = int(train_count)
        trainX = trainX[(t_row-train_count):t_row]
        #train_percentage.set(trainX.shape[0])
        #trainX = np.asarray(trainX)
    elif radSel == 2:
        trainX = trainX[:t_perc]
        
    trainX = np.asarray(trainX)
    
    return trainX
        
def count_test():
    
    global testY
   
    targets = listTarget.get(0, END)
    targets = np.asarray(targets)
    testY = df_test.get(targets)      
    testY = np.asarray(testY)
    
def difference(data, interval):
    return [data[i] - data[i - interval] for i in range(interval, len(data))]



def create_model():
    
    global model
    global trainX,trainY
    global scaler
    predictor = get_Predictor()
    target = get_Target()
    

    predictor = np.asarray(predictor)
    target = np.asarray(target)
    
    
    
    differenceType = int(varDifference.get())
    
    if differenceType == 1:
        interval = int(entry_ınterval.get())
        target = difference(target, interval)
        target = np.asarray(target)
        predictor = difference(predictor,interval)
        predictor = np.asarray(predictor)
        
        
    normalizationType = comboScaler.get()
    if normalizationType == 'Min Max Normalization':
        normalizationType = 1
        
    if normalizationType == 'Standardization':
        normalizationType = 2
    
    if normalizationType == 1:
        scaler = MinMaxScaler()
        predictor = scaler.fit_transform(predictor)
        target = scaler.fit_transform(target)
        
    if normalizationType == 2:
        scaler = StandardScaler()
        predictor = scaler.fit_transform(predictor)
        target = scaler.fit_transform(target)
    
    dataset = np.hstack((predictor,target))
    
    
    
    lookback = int(lag_num.get())
    
    trainX, trainY = create_dataset(dataset, lookback)
    

    
    number_hid = int(num_hid.get())
    lr = float(n_rate.get())
    moment = float(n_momentum.get())
    epoch = int(n_epoch.get())
    batchSize = int(n_batch.get())
    
    if lr > 0:
        if moment > 0:
            SGD = optimizers.SGD(learning_rate = lr,momentum = moment)
        else:
            SGD = optimizers.SGD(learning_rate = lr)
            RMSProb = optimizers.RMSprop(learning_rate = lr)
            Adamax = optimizers.Adamax(learning_rate = lr)
            Adadelta = optimizers.Adadelta(learning_rate = lr)
            Adam = optimizers.Adam(learning_rate = lr)
            Adagrad = optimizers.Adagrad(learning_rate = lr)
    else:
        SGD = optimizers.SGD()
        RMSProb = optimizers.RMSprop()
        Adamax = optimizers.Adamax()
        Adadelta = optimizers.Adadelta()
        Adam = optimizers.Adam()
        Adagrad = optimizers.Adagrad()
        
        
    modelType = int(radVar_modelType.get())
    
    
   ################################################### Create MLP Model  ##################################################
                     
    if modelType == 1:

        n_input = trainX.shape[1] * trainX.shape[2]
        trainX = trainX.reshape((trainX.shape[0], n_input))
            
            
        if number_hid == 1:
                
            model = Sequential()
            model.add(Dense(int(entry_first_layer.get()),input_dim=n_input, activation=comboFirstLayer.get()))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
            model.summary()
            
            loss = history.history['loss']
                
            trainError.set(loss[epoch-1])
                
        elif number_hid == 2:
                
            model = Sequential()
            model.add(Dense(int(entry_first_layer.get()),input_dim=n_input, activation=comboFirstLayer.get()))
            model.add(Dense(int(entry_second_layer.get()), activation=comboSecondLayer.get()))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)  
            model.summary()
                
            loss = history.history['loss']
            
            trainError.set(loss[epoch-1])
                
        elif number_hid == 3:   
                
            model = Sequential()
            model.add(Dense(int(entry_first_layer.get()),input_dim=n_input, activation=comboFirstLayer.get()))
            model.add(Dense(int(entry_second_layer.get()), activation=comboSecondLayer.get()))
            model.add(Dense(int(entry_third_layer.get()), activation=comboThirdLayer.get()))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
            model.summary()
                
            loss = history.history['loss']
                
            trainError.set(loss[epoch-1])
                
                
                
        minLoss = 2**31
        bestFirstLayer = 0
        bestSecondLayer = 0
        bestThirdLayer = 0
                
        if number_hid == 4:
            for x in range(int(n_min.get()),int(n_max.get())+1):
                    
                model = Sequential()
                model.add(Dense(x,input_dim=n_input, activation='relu'))
                model.add(Dense(1))
                model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
                model.summary()   
                    
                loss = history.history['loss']
                finalLoss = loss[epoch-1]
                    
                    
                if finalLoss < minLoss:
                    minLoss = finalLoss
                    bestFirstLayer = x
                
            trainError.set(minLoss)
            firstLayer.set(bestFirstLayer)   
            
            model = Sequential()
            model.add(Dense(bestFirstLayer,input_dim=n_input, activation='relu'))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
            
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
            model.summary()   
                    
                    
        elif number_hid == 5:
            for x in range(int(n21_min.get()),int(n21_max.get())+1):
                for y in range(int(n22_min.get()),int(n22_max.get())+1):
                        
                    model = Sequential()
                    model.add(Dense(x,input_dim=n_input, activation=comboFirstLayer.get()))
                    model.add(Dense(y, activation=comboSecondLayer.get()))
                    model.add(Dense(1))
                    model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                    history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
                    model.summary()
                       
                       
                    loss = history.history['loss']
                    finalLoss = loss[epoch-1]
                       
                       
                    if finalLoss < minLoss:
                       minLoss = finalLoss
                       bestFirstLayer = x
                       bestSecondLayer = y
            
            trainError.set(minLoss)
            firstLayer.set(bestFirstLayer)
            secondLayer.set(bestSecondLayer)
            
            model = Sequential()
            model.add(Dense(bestFirstLayer,input_dim=n_input, activation=comboFirstLayer.get()))
            model.add(Dense(bestSecondLayer, activation=comboSecondLayer.get()))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
            model.summary()
                          
                       
        elif number_hid == 6:
            for x in range(int(n31_min.get()),int(n31_max.get())+1):
                for y in range(int(n32_min.get()),int(n32_max.get())+1):
                    for z in range(int(n33_min.get()),int(n33_max.get())+1):
                            
                        model = Sequential()
                        model.add(Dense(x,input_dim=n_input, activation=comboFirstLayer.get()))
                        model.add(Dense(y, activation=comboSecondLayer.get()))
                        model.add(Dense(z, activation=comboThirdLayer.get()))
                        model.add(Dense(1))
                        model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                        history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
                        model.summary()
                             
                        loss = history.history['loss']
                        finalLoss = loss[epoch-1]
                            
                             
                        if finalLoss < minLoss:
                            minLoss = finalLoss
                            bestFirstLayer = x
                            bestSecondLayer = y
                            bestThirdLayer = z
                                 
            trainError.set(minLoss)
            firstLayer.set(bestFirstLayer)
            secondLayer.set(bestSecondLayer)
            thirdLayer.set(bestThirdLayer)
            
            model = Sequential()
            model.add(Dense(bestFirstLayer,input_dim=n_input, activation=comboFirstLayer.get()))
            model.add(Dense(bestSecondLayer, activation=comboSecondLayer.get()))
            model.add(Dense(bestThirdLayer, activation=comboThirdLayer.get()))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
            model.summary()
            
            
    ################################################### Create LSTM Model  ##################################################
            
    if modelType == 2:

        n_features = int(trainX.shape[2])
        n_steps = int(trainX.shape[1])
        trainX = trainX.reshape((trainX.shape[0], n_steps , n_features))
            
        if number_hid == 1:
                
            model = Sequential()
            model.add(LSTM(int(entry_first_layer.get()),input_shape=(n_steps,n_features), activation=comboFirstLayer.get()))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
            model.summary()
            
            loss = history.history['loss']
                
            trainError.set(loss[epoch-1])
                
        elif number_hid == 2:
                
            model = Sequential()
            model.add(LSTM(int(entry_first_layer.get()),input_shape=(n_steps,n_features), activation=comboFirstLayer.get(),return_sequences=True))
            model.add(LSTM(int(entry_second_layer.get()), activation=comboSecondLayer.get()))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)  
            model.summary()
                
            loss = history.history['loss']
            
            trainError.set(loss[epoch-1])
                
        elif number_hid == 3:   
                
            model = Sequential()
            model.add(LSTM(int(entry_first_layer.get()),input_shape=(n_steps,n_features), activation=comboFirstLayer.get(),return_sequences=True))
            model.add(LSTM(int(entry_second_layer.get()), activation=comboSecondLayer.get(),return_sequences=True))
            model.add(LSTM(int(entry_third_layer.get()), activation=comboThirdLayer.get()))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
            
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
            model.summary()
                
            loss = history.history['loss']
                
            trainError.set(loss[epoch-1])
                
                
                
        minLoss = 2**31
        bestFirstLayer = 0
        bestSecondLayer = 0
        bestThirdLayer = 0
                
        if number_hid == 4:
            
            for x in range(int(n_min.get()),int(n_max.get())+1):
                    
                model = Sequential()
                model.add(LSTM(x,input_shape=(n_steps, n_features), activation=comboFirstLayer.get()))
                model.add(Dense(1))
                model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                
                history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
                model.summary()   
                    
                loss = history.history['loss']
                finalLoss = loss[epoch-1]
                    
                    
                if finalLoss < minLoss:
                    minLoss = finalLoss
                    bestFirstLayer = x
                
            trainError.set(minLoss)
            firstLayer.set(bestFirstLayer) 
            
            model = Sequential()
            model.add(LSTM(bestFirstLayer,input_shape=(n_steps, n_features), activation=comboFirstLayer.get()))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
            model.summary()
                    
                    
        elif number_hid == 5:
            for x in range(int(n21_min.get()),int(n21_max.get())+1):
                for y in range(int(n22_min.get()),int(n22_max.get())+1):
                        
                    model = Sequential()
                    model.add(LSTM(x,input_shape=(n_steps, n_features), activation=comboFirstLayer.get(),return_sequences = True))
                    model.add(LSTM(y, activation=comboSecondLayer.get()))
                    model.add(Dense(1))
                    model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                    
                    history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
                    
                    model.summary()
                       
                       
                    loss = history.history['loss']
                    finalLoss = loss[epoch-1]
                       
                       
                    if finalLoss < minLoss:
                       minLoss = finalLoss
                       bestFirstLayer = x
                       bestSecondLayer = y
            
            trainError.set(minLoss)
            firstLayer.set(bestFirstLayer)
            secondLayer.set(bestSecondLayer)
            
            model = Sequential()
            model.add(LSTM(bestFirstLayer,input_shape=(n_steps, n_features), activation=comboFirstLayer.get(),return_sequences = True))
            model.add(LSTM(bestSecondLayer, activation=comboSecondLayer.get()))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                    
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
                    
            model.summary()
                          
                       
        elif number_hid == 6:
            for x in range(int(n31_min.get()),int(n31_max.get())+1):
                for y in range(int(n32_min.get()),int(n32_max.get())+1):
                    for z in range(int(n33_min.get()),int(n33_max.get())+1):
                            
                        model = Sequential()
                        model.add(LSTM(x,input_shape=(n_steps, n_features), activation=comboFirstLayer.get(),return_sequences = True))
                        model.add(LSTM(y, activation=comboSecondLayer.get(),return_sequences = True))
                        model.add(LSTM(z, activation=comboThirdLayer.get()))
                        model.add(Dense(1))
                        model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                        history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
                        model.summary()
                             
                        loss = history.history['loss']
                        finalLoss = loss[epoch-1]
                            
                             
                        if finalLoss < minLoss:
                            minLoss = finalLoss
                            bestFirstLayer = x
                            bestSecondLayer = y
                            bestThirdLayer = z
                                 
            trainError.set(minLoss)
            firstLayer.set(bestFirstLayer)
            secondLayer.set(bestSecondLayer)
            thirdLayer.set(bestThirdLayer)
            
            model = Sequential()
            model.add(LSTM(bestFirstLayer,input_shape=(n_steps, n_features), activation=comboFirstLayer.get(),return_sequences = True))
            model.add(LSTM(bestSecondLayer, activation=comboSecondLayer.get(),return_sequences = True))
            model.add(LSTM(bestThirdLayer, activation=comboThirdLayer.get()))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
            model.summary()


################################################ Create Bidirectional LSTM Model ############################################################

    if modelType == 3:

        n_features = int(trainX.shape[2])
        n_steps = int(trainX.shape[1])
        trainX = trainX.reshape((trainX.shape[0], n_steps , n_features))
            
        if number_hid == 1:
                
            model = Sequential()
            model.add(Bidirectional(LSTM(int(entry_first_layer.get()),input_shape=(n_steps,n_features), activation=comboFirstLayer.get())))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
            model.summary()
            
            loss = history.history['loss']
                
            trainError.set(loss[epoch-1])
                
        elif number_hid == 2:
                
            model = Sequential()
            model.add(Bidirectional(LSTM(int(entry_first_layer.get()),input_shape=(n_steps,n_features), activation=comboFirstLayer.get(),return_sequences=True)))
            model.add(Bidirectional(LSTM(int(entry_second_layer.get()), activation=comboSecondLayer.get())))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)  
            model.summary()
                
            loss = history.history['loss']
            
            trainError.set(loss[epoch-1])
                
        elif number_hid == 3:   
                
            model = Sequential()
            model.add(Bidirectional(LSTM(int(entry_first_layer.get()),input_shape=(n_steps,n_features), activation=comboFirstLayer.get(),return_sequences=True)))
            model.add(Bidirectional(LSTM(int(entry_second_layer.get()), activation=comboSecondLayer.get(),return_sequences=True)))
            model.add(Bidirectional(LSTM(int(entry_third_layer.get()), activation=comboThirdLayer.get())))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
            
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
            model.summary()
                
            loss = history.history['loss']
                
            trainError.set(loss[epoch-1])
                
                
                
        minLoss = 2**31
        bestFirstLayer = 0
        bestSecondLayer = 0
        bestThirdLayer = 0
                
        if number_hid == 4:
            
            for x in range(int(n_min.get()),int(n_max.get())+1):
                    
                model = Sequential()
                model.add(Bidirectional(LSTM(x,input_shape=(n_steps, n_features), activation=comboFirstLayer.get())))
                model.add(Dense(1))
                model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                
                history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
                model.summary()   
                    
                loss = history.history['loss']
                finalLoss = loss[epoch-1]
                    
                    
                if finalLoss < minLoss:
                    minLoss = finalLoss
                    bestFirstLayer = x
                
            trainError.set(minLoss)
            firstLayer.set(bestFirstLayer) 
            
            model = Sequential()
            model.add(Bidirectional(LSTM(bestFirstLayer,input_shape=(n_steps, n_features), activation=comboFirstLayer.get())))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
            model.summary()  
                    
                    
        elif number_hid == 5:
            for x in range(int(n21_min.get()),int(n21_max.get())+1):
                for y in range(int(n22_min.get()),int(n22_max.get())+1):
                        
                    model = Sequential()
                    model.add(Bidirectional(LSTM(x,input_shape=(n_steps, n_features), activation=comboFirstLayer.get(),return_sequences = True)))
                    model.add(Bidirectional(LSTM(y, activation=comboSecondLayer.get())))
                    model.add(Dense(1))
                    model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                    
                    history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
                    
                    model.summary()
                       
                       
                    loss = history.history['loss']
                    finalLoss = loss[epoch-1]
                       
                       
                    if finalLoss < minLoss:
                       minLoss = finalLoss
                       bestFirstLayer = x
                       bestSecondLayer = y
            
            trainError.set(minLoss)
            firstLayer.set(bestFirstLayer)
            secondLayer.set(bestSecondLayer)
            
            model = Sequential()
            model.add(Bidirectional(LSTM(bestFirstLayer,input_shape=(n_steps, n_features), activation=comboFirstLayer.get(),return_sequences = True)))
            model.add(Bidirectional(LSTM(bestSecondLayer, activation=comboSecondLayer.get())))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                    
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
                          
                       
        elif number_hid == 6:
            for x in range(int(n31_min.get()),int(n31_max.get())+1):
                for y in range(int(n32_min.get()),int(n32_max.get())+1):
                    for z in range(int(n33_min.get()),int(n33_max.get())+1):
                            
                        model = Sequential()
                        model.add(Bidirectional(LSTM(x,input_shape=(n_steps, n_features), activation=comboFirstLayer.get(),return_sequences = True)))
                        model.add(Bidirectional(LSTM(y, activation=comboSecondLayer.get(),return_sequences = True)))
                        model.add(Bidirectional(LSTM(z, activation=comboThirdLayer.get())))
                        model.add(Dense(1))
                        model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
                        history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
                        model.summary()
                             
                        loss = history.history['loss']
                        finalLoss = loss[epoch-1]
                            
                             
                        if finalLoss < minLoss:
                            minLoss = finalLoss
                            bestFirstLayer = x
                            bestSecondLayer = y
                            bestThirdLayer = z
                                 
            trainError.set(minLoss)
            firstLayer.set(bestFirstLayer)
            secondLayer.set(bestSecondLayer)
            thirdLayer.set(bestThirdLayer)
            
            model = Sequential()
            model.add(Bidirectional(LSTM(bestFirstLayer,input_shape=(n_steps, n_features), activation=comboFirstLayer.get(),return_sequences = True)))
            model.add(Bidirectional(LSTM(bestSecondLayer, activation=comboSecondLayer.get(),return_sequences = True)))
            model.add(Bidirectional(LSTM(bestThirdLayer, activation=comboThirdLayer.get())))
            model.add(Dense(1))
            model.compile(loss=comboLoss.get(), optimizer=comboOptimizer.get())
            history = model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=2)
            model.summary()


def test_model():
    
    global actualValues
    global values
    
    predictor = get_Predictor()
    target = get_Target()
    

    predictor = np.asarray(predictor)
    target = np.asarray(target)
      

    differenceType = int(varDifference.get())
    
    if differenceType == 1:
        interval = int(entry_ınterval.get())
        target = difference(target, interval)
        target = np.asarray(target)
        predictor = difference(predictor,interval)
        predictor = np.asarray(predictor)
        

    normalizationType = comboScaler.get()
    
    if normalizationType == 'Min Max Normalization':
        normalizationType = 1
        
    if normalizationType == 'Standardization':
        normalizationType = 2
    
    
    if normalizationType == 1:
        scaler = MinMaxScaler()
        predictor = scaler.fit_transform(predictor)
        target = scaler.fit_transform(target)
        
    if normalizationType == 2:
        scaler = StandardScaler()
        predictor = scaler.fit_transform(predictor)
        target = scaler.fit_transform(target)
    
    dataset = np.hstack((predictor,target))
    
    
    lookback = int(lag_num.get())
    
    trainX, trainY = create_dataset(dataset, lookback)
    
    a = trainX[-1]
    
    
    values = []
    forecast_num = int(number_forecast.get())
    modelType = int(radVar_modelType.get())
    
    if modelType == 1:
    
        a = a.reshape(1,a.shape[0])
        
        for i in range(0,forecast_num):
            yhat = model.predict(a, verbose=0)      
            values = np.append(values,yhat)
            a = np.append(a[:,1:],yhat)
            a = a.reshape(1,a.shape[0])
            
            
    if modelType == 2 or modelType == 3:
        
        n_steps = int(trainX.shape[1])
        n_features = trainX.shape[2]
        a = a.reshape((1, n_steps, n_features))
        
        for i in range(0,forecast_num):
            yhat = model.predict(a, verbose=0)      
            print(yhat.shape)
            values = np.append(values,yhat)
            a = a.reshape((a.shape[0], a.shape[1] * a.shape[2]))
            a = np.append(a[:,1:],yhat)
            a = a.reshape(1,a.shape[0])
            a = a.reshape((1, n_steps, n_features))
            print(values.shape)

    count_test()

    values = values.reshape(-1,1)
    print(values.shape)

    predictor = get_Predictor()
    target = get_Target()

    predictor = np.asarray(predictor)
    target = np.asarray(target)
    
    if normalizationType == 1 or normalizationType == 2:
        values = scaler.inverse_transform(values)
    else:
        values = values
        
        
        
    if differenceType == 1:
        interval = int(entry_ınterval.get())
        for i in range (0,len(values)):
            if(interval<=i):
                values[i] = values[i] + values[i-interval]
            else:
                values[i] = values[i] + target[(len(target)-interval)+i]
    
    actualValues = testY[:forecast_num]
    testScore = mean_absolute_percentage_error(actualValues,values)
    
    test_mape.set(testScore)

    

    
def create_dataset(sequences, n_steps = 1):
    
    X, y = list(), list()
    for i in range(len(sequences)):
		# find the end of this pattern
        end_ix = i + n_steps
		# check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        #gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
        

    lag_type = rad_lag.get()
    
    X = np.array(X)
    y = np.array(y)
    
    acf_vals = acf(y, nlags=n_steps, fft=False)
    
    if lag_type == 2:
       
        lag = selected_lag.get()
        li = list(lag.split(","))
        li = [int(i) for i in li] 
        X = X[:,li,:] 
        
    elif lag_type == 3:
        
        best_n = int(best_lag.get())
        acf_vals = acf_vals[1:]
        acf_vals = np.argsort(acf_vals)
        acf_vals = acf_vals[::-1]
        acf_vals = acf_vals[:best_n]
        X = X[:,acf_vals,:]
        
    elif lag_type == 4:
        
        acf_vals = acf_vals[1:]
        threshold = float(bigger_lag.get())
        a = [i for i,v in enumerate(acf_vals) if v > threshold]
        print(a)
        X = X[:,a,:]
        
    return np.array(X), np.array(y)


def model_save():
    directory = filedialog.asksaveasfile(mode='w',defaultextension='.h5')
    model.save(directory.name)
    

def enable_model_browser():
    model_button.configure(state='normal')
    entry_model.configure(state='normal')
    
def disable_model_browser():
    model_button.configure(state='disabled')
    entry_model.configure(state='disabled')
    
def import_model():
    global v_model
    global model
    model_file_path = filedialog.askopenfilename(filetypes=(("H5 Files", "*.h5"), ("All", "*.*")))
    v_model.set(model_file_path)
    model = load_model(model_file_path)
    
def show_acf():
    radSel = radVar_train.get()
    t_perc = int(train_percentage.get())
    targets = listTarget.get(0, END)
    targets = np.asarray(targets)
    trainX = df_train.get(targets)
    t_row = trainX.shape[0]
    if radSel == 1:
        train_count = (t_perc/100)*t_row
        train_count = int(train_count)
        trainX = trainX[(t_row-train_count):t_row]
        #train_percentage.set(trainX.shape[0])
        #trainX = np.asarray(trainX)
    elif radSel == 2:
        trainX = trainX[:t_perc]
        
    trainX = pd.DataFrame(trainX)
    fig, axes = plt.subplots(1,2,figsize=(16,3), dpi=100)
    plot_acf(trainX.values.tolist(), lags=lag_num.get(), ax=axes[0])
    plot_pacf(trainX.values.tolist(), lags=lag_num.get(), ax=axes[1])
    
    root2 = tk.Toplevel()
    canvas = FigureCanvasTkAgg(fig, master=root2)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def enable_selected_lag():
    entry_selected_lag.configure(state='normal')
    entry_best_lag.configure(state='disabled')
    entry_bigger_lag.configure(state='disabled')
    
def enable_best_lag():
    entry_best_lag.configure(state='normal')
    entry_selected_lag.configure(state='disabled')
    entry_bigger_lag.configure(state='disabled')
    
def enable_bigger_lag():
    entry_bigger_lag.configure(state='normal')
    entry_selected_lag.configure(state='disabled')
    entry_best_lag.configure(state='disabled')

def enable_first():
    entry_first_layer.configure(state='normal')
    entry_second_layer.configure(state='disabled')
    entry_third_layer.configure(state='disabled')
    
    entry_n_min.configure(state='disabled')
    entry_n_max.configure(state='disabled')
    entry_n21_min.configure(state='disabled')
    entry_n21_max.configure(state='disabled')
    entry_n22_min.configure(state='disabled')
    entry_n22_max.configure(state='disabled')
    entry_n31_min.configure(state='disabled')
    entry_n31_max.configure(state='disabled')
    entry_n32_min.configure(state='disabled')
    entry_n32_max.configure(state='disabled')
    entry_n33_min.configure(state='disabled')
    entry_n33_max.configure(state='disabled')
    
    comboFirstLayer.configure(state='normal')
    comboSecondLayer.configure(state='disabled')
    comboThirdLayer.configure(state='disabled')

    
def enable_second():
    entry_first_layer.configure(state='normal')
    entry_second_layer.configure(state='normal')
    entry_third_layer.configure(state='disabled')
    
    entry_n_min.configure(state='disabled')
    entry_n_max.configure(state='disabled')
    entry_n21_min.configure(state='disabled')
    entry_n21_max.configure(state='disabled')
    entry_n22_min.configure(state='disabled')
    entry_n22_max.configure(state='disabled')
    entry_n31_min.configure(state='disabled')
    entry_n31_max.configure(state='disabled')
    entry_n32_min.configure(state='disabled')
    entry_n32_max.configure(state='disabled')
    entry_n33_min.configure(state='disabled')
    entry_n33_max.configure(state='disabled')
    
    comboFirstLayer.configure(state='normal')
    comboSecondLayer.configure(state='normal')
    comboThirdLayer.configure(state='disabled')

def enable_third():
    entry_first_layer.configure(state='normal')
    entry_second_layer.configure(state='normal')
    entry_third_layer.configure(state='normal')
    
    entry_n_min.configure(state='disabled')
    entry_n_max.configure(state='disabled')
    entry_n21_min.configure(state='disabled')
    entry_n21_max.configure(state='disabled')
    entry_n22_min.configure(state='disabled')
    entry_n22_max.configure(state='disabled')
    entry_n31_min.configure(state='disabled')
    entry_n31_max.configure(state='disabled')
    entry_n32_min.configure(state='disabled')
    entry_n32_max.configure(state='disabled')
    entry_n33_min.configure(state='disabled')
    entry_n33_max.configure(state='disabled')
    
    comboFirstLayer.configure(state='normal')
    comboSecondLayer.configure(state='normal')
    comboThirdLayer.configure(state='normal')
    
  
    
def enable_oto_one():
    entry_n_min.configure(state='normal')
    entry_n_max.configure(state='normal')
    entry_n21_min.configure(state='disabled')
    entry_n21_max.configure(state='disabled')
    entry_n22_min.configure(state='disabled')
    entry_n22_max.configure(state='disabled')
    entry_n31_min.configure(state='disabled')
    entry_n31_max.configure(state='disabled')
    entry_n32_min.configure(state='disabled')
    entry_n32_max.configure(state='disabled')
    entry_n33_min.configure(state='disabled')
    entry_n33_max.configure(state='disabled')
    
    entry_first_layer.configure(state='disabled')
    entry_second_layer.configure(state='disabled')
    entry_third_layer.configure(state='disabled')
    
    comboFirstLayer.configure(state='disabled')
    comboSecondLayer.configure(state='disabled')
    comboThirdLayer.configure(state='disabled')
    

    
def enable_oto_two():
    entry_n_min.configure(state='disabled')
    entry_n_max.configure(state='disabled')
    entry_n21_min.configure(state='normal')
    entry_n21_max.configure(state='normal')
    entry_n22_min.configure(state='normal')
    entry_n22_max.configure(state='normal')
    entry_n31_min.configure(state='disabled')
    entry_n31_max.configure(state='disabled')
    entry_n32_min.configure(state='disabled')
    entry_n32_max.configure(state='disabled')
    entry_n33_min.configure(state='disabled')
    entry_n33_max.configure(state='disabled')
    
    entry_first_layer.configure(state='disabled')
    entry_second_layer.configure(state='disabled')
    entry_third_layer.configure(state='disabled')
    
    comboFirstLayer.configure(state='disabled')
    comboSecondLayer.configure(state='disabled')
    comboThirdLayer.configure(state='disabled')
    

    
def enable_oto_three():
    entry_n_min.configure(state='disabled')
    entry_n_max.configure(state='disabled')
    entry_n21_min.configure(state='disabled')
    entry_n21_max.configure(state='disabled')
    entry_n22_min.configure(state='disabled')
    entry_n22_max.configure(state='disabled')
    entry_n31_min.configure(state='normal')
    entry_n31_max.configure(state='normal')
    entry_n32_min.configure(state='normal')
    entry_n32_max.configure(state='normal')
    entry_n33_min.configure(state='normal')
    entry_n33_max.configure(state='normal')
    
    entry_first_layer.configure(state='disabled')
    entry_second_layer.configure(state='disabled')
    entry_third_layer.configure(state='disabled')
    
    comboFirstLayer.configure(state='disabled')
    comboSecondLayer.configure(state='disabled')
    comboThirdLayer.configure(state='disabled')
    
def showActualvsForecasted():
    
    fig = plt.figure(figsize=(9, 3))

    plt.plot(values, color = 'red', label ='Forecasted Values')
    plt.plot(actualValues, color='blue', label = 'Actual Values')
    plt.title('actual vs forecasted')

    plt.ylabel('value')
    plt.xlabel('forecast')
    
    root3 = tk.Toplevel()
    canvas = FigureCanvasTkAgg(fig, master=root3)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
def show_forecasted_vals():
    
    window = tk.Toplevel(root)
    window.geometry('300x500')
    
    global actualValues
    global values
    
    predictor = get_Predictor()
    target = get_Target()
    

    predictor = np.asarray(predictor)
    target = np.asarray(target)
      

    differenceType = int(varDifference.get())
    
    if differenceType == 1:
        interval = int(entry_ınterval.get())
        target = difference(target, interval)
        target = np.asarray(target)
        predictor = difference(predictor,interval)
        predictor = np.asarray(predictor)
        

    normalizationType = comboScaler.get()
    
    if normalizationType == 'Min Max Normalization':
        normalizationType = 1
        
    if normalizationType == 'Standardization':
        normalizationType = 2
    
    
    if normalizationType == 1:
        scaler = MinMaxScaler()
        predictor = scaler.fit_transform(predictor)
        target = scaler.fit_transform(target)
        
    if normalizationType == 2:
        scaler = StandardScaler()
        predictor = scaler.fit_transform(predictor)
        target = scaler.fit_transform(target)
    
    dataset = np.hstack((predictor,target))
    
    
    lookback = int(lag_num.get())
    
    trainX, trainY = create_dataset(dataset, lookback)
    
    a = trainX[-1]
    
    
    values = []
    forecast_num = int(number_forecast.get())
    modelType = int(radVar_modelType.get())
    
    if modelType == 1:
    
        a = a.reshape(1,a.shape[0])
        
        for i in range(0,forecast_num):
            yhat = model.predict(a, verbose=0)      
            values = np.append(values,yhat)
            a = np.append(a[:,1:],yhat)
            a = a.reshape(1,a.shape[0])
            
            
    if modelType == 2 or modelType == 3:
        
        n_steps = int(trainX.shape[1])
        n_features = trainX.shape[2]
        a = a.reshape((1, n_steps, n_features))
        
        for i in range(0,forecast_num):
            yhat = model.predict(a, verbose=0)      
            values = np.append(values,yhat)
            a = a.reshape((a.shape[0], a.shape[1] * a.shape[2]))
            a = np.append(a[:,1:],yhat)
            a = a.reshape(1,a.shape[0])
            a = a.reshape((1, n_steps, n_features))




    values = values.reshape(-1,1)
    
    predictor = get_Predictor()
    target = get_Target()

    predictor = np.asarray(predictor)
    target = np.asarray(target)
    
    if normalizationType == 1 or normalizationType == 2:
        values = scaler.inverse_transform(values)
    else:
        values = values
        
        
        
    if differenceType == 1:
        interval = int(entry_ınterval.get())
        for i in range (0,len(values)):
            if(interval<=i):
                values[i] = values[i] + values[i-interval]
            else:
                values[i] = values[i] + target[(len(target)-interval)+i]
    
    
    listForecastVals = Listbox(window)
    listForecastVals.grid(row=0, column=0)
    
    for i in range (0,forecast_num):
        listForecastVals.insert(i,values[i])
    
    
    
    tk.Button(window, text ="Save Forecast Values",command=save_forecast_results).grid(row=2,column=0,sticky='w')
    
    
def save_forecast_results():
    directory = filedialog.asksaveasfile(mode='w',defaultextension='.xlsx')
    workbook = xlsxwriter.Workbook(directory.name) 
    print(directory.name)
    worksheet = workbook.add_worksheet() 
    
    row = 0
    column = 0
    
    for i in range (0,len(values)):
        worksheet.write(row,column,i)
        column+=1
        worksheet.write(row,column,values[i])
        row+=1
        column=0
        
    workbook.close()
    
def enable_interval():
        
    value = int(varDifference.get())
    if value == 0:
        entry_ınterval.configure(state='disabled')
    else:
        entry_ınterval.configure(state='normal')


    

####################################################################################
    
root = tk.Tk()
#w, h = root.winfo_screenwidth(), root.winfo_screenheight()
#root.overrideredirect(1)
#root.geometry("%dx%d+0+0" % (w, h))
root.geometry('1350x765')


train_frame = ttk.Labelframe(root,text = 'Get Train Set')
train_frame.grid(row=0,column=0,rowspan=2,columnspan=2,padx=5,sticky='w')

#Train Set Browser
tk.Label(train_frame, text='Train File Path').grid(row=0, column=0,sticky='w')
v_train = tk.StringVar()
entry_train = tk.Entry(train_frame, textvariable=v_train).grid(row=0, column=1,sticky='w')
tk.Button(train_frame, text='Browse Train Set', command=import_train_data).grid(row=0, column=2)
tk.Button(train_frame, text='Read CSV', command=lambda: read_csv_file()).grid(row=0, column=3,sticky='w')

#Column Listbox
listAll = Listbox(train_frame)
listAll.grid(row=1, column=0)


#Predictor Listbox
listPredictor = Listbox(train_frame)
listPredictor.grid(row=1, column=1)
tk.Button(train_frame, text="Add Predictor", command=lambda: get_active_delete(listPredictor)).grid(row=2, column=1)
tk.Button(train_frame, text="Eject Predictor", command=lambda: eject_active_delete(listPredictor)).grid(row=3, column=1)

#Target Listbox
listTarget = Listbox(train_frame)
listTarget.grid(row=1, column=2)
tk.Button(train_frame, text="Add Target", command=lambda: get_active_delete(listTarget)).grid(row=2, column=2)
tk.Button(train_frame, text="Eject Target", command=lambda: eject_active_delete(listTarget)).grid(row=3, column=2)


########################### TRAIN COSTUMIZE FRAME #############################################

train_costumize_frame = ttk.Labelframe(root,text = 'Customize Train Set')
train_costumize_frame.grid(row=2,column=0,padx = 5)

ttk.Label(train_costumize_frame, text='# of Row in Train Set').grid(row=0, column=0,pady = 5)
train_percentage = tk.StringVar()
entry_train_percent = tk.Entry(train_costumize_frame, textvariable=train_percentage).grid(row=1, column=0, pady=5)
#tk.Button(train_costumize_frame ,text="Count Rows",command=get_Target).grid(row=1, column=2)

radVar_train = tk.IntVar()
rad1 = tk.Radiobutton(train_costumize_frame, text = 'As Percent',variable = radVar_train, value = 1).grid(row = 0, column = 1)
rad2 = tk.Radiobutton(train_costumize_frame, text = 'As Number',variable = radVar_train, value = 2).grid(row = 0, column = 2)
'''
radVar_scaler = tk.IntVar()
rad1_Scale = tk.Radiobutton(train_costumize_frame, text = 'Min Max Normalization',variable = radVar_scaler, value = 1).grid(row = 2, column = 0)
rad2_Scale = tk.Radiobutton(train_costumize_frame, text = 'Standardization',variable = radVar_scaler, value = 2).grid(row = 2, column = 1)
'''

ttk.Label(train_costumize_frame, text='Scale Type').grid(row=2, column=0,padx=5)

comboScaler = ttk.Combobox(train_costumize_frame, 
                            values=[
                                    'None', 
                                    'Min Max Normalization',
                                    'Standardization' 
                                    ])

comboScaler.grid(column=1, row=2)
comboScaler.current(0)

varDifference = tk.IntVar()
cbDifference = tk.Checkbutton(train_costumize_frame, text='Difference',variable=varDifference, onvalue=1, offvalue=0, command=enable_interval)
cbDifference.grid(row=3,column=0,sticky='w')


ttk.Label(train_costumize_frame, text='Interval').grid(row=3, column=1,columnspan=2,sticky='w')
ıntervalEntry = tk.StringVar()
entry_ınterval = tk.Entry(train_costumize_frame, textvariable=ıntervalEntry,state='disabled')
entry_ınterval.grid(row=3, column=2)

############################## LAG OPTIONS FRAME #########################################

lag_options_frame = ttk.Labelframe(root,text = 'Lag Options')
lag_options_frame.grid(row=3,column=0,rowspan=2,columnspan=2,sticky='w',pady=10,padx=5)

#Number of Lags
ttk.Label(lag_options_frame, text='Number of Lags').grid(row=0, column=0,sticky='w')
lag_num = tk.StringVar()
entry_lag_num = tk.Entry(lag_options_frame, textvariable=lag_num).grid(row=0, column=1,sticky='w')

tk.Button(lag_options_frame, text ="Show ACF Graph",command=show_acf).grid(row=0,column=2,sticky='w')

rad_lag = tk.IntVar()
rad_lag1 = tk.Radiobutton(lag_options_frame, text = 'Use All Lags',variable = rad_lag, value = 1).grid(row = 1, column = 0,sticky='w')

rad_lag2 = tk.Radiobutton(lag_options_frame, text = 'Use Selected(1,3,..)',variable = rad_lag, value = 2,command = enable_selected_lag).grid(row = 2, column = 0,sticky='w')
selected_lag = tk.StringVar()
entry_selected_lag = tk.Entry(lag_options_frame, textvariable=selected_lag, state = 'disabled')
entry_selected_lag.grid(row=2, column=1, pady=5,sticky='w')

rad_lag3 = tk.Radiobutton(lag_options_frame, text = 'Use Best N',variable = rad_lag, value = 3, command = enable_best_lag).grid(row = 3, column = 0,sticky='w')
best_lag = tk.StringVar()
entry_best_lag = tk.Entry(lag_options_frame, textvariable=best_lag,state = 'disabled')
entry_best_lag.grid(row=3, column=1, pady=5,sticky='w')

rad_lag4 = tk.Radiobutton(lag_options_frame, text = 'Use Correllation > n',variable = rad_lag, value = 4, command = enable_bigger_lag).grid(row = 4, column = 0,sticky='w')
bigger_lag = tk.StringVar()
entry_bigger_lag = tk.Entry(lag_options_frame, textvariable=bigger_lag,state = 'disabled')
entry_bigger_lag.grid(row=4, column=1, pady=5,sticky='w')




##### MODEL FRAME ########################################## 

### MODEL WITHOUT OPTIMIZATION #####

model_frame = ttk.Labelframe(root,text = 'Create Model')
model_frame.grid(row=0,column=1,rowspan=3,columnspan=3,sticky='W',padx=60)

model_without_optimization = ttk.Labelframe(model_frame, text = 'Model Without Optimization')
model_without_optimization.grid(row=0,column=0,sticky='W')

ttk.Label(model_without_optimization, text='Number of Hidden Layer').grid(row=0, column=0)
num_hid = tk.IntVar()

num_hid1 = tk.Radiobutton(model_without_optimization, text = '1',variable = num_hid, value = 1,command = enable_first)
num_hid1.grid(row = 0, column = 1)

num_hid2 = tk.Radiobutton(model_without_optimization, text = '2',variable = num_hid, value = 2,command = enable_second)
num_hid2.grid(row = 0, column = 2)

num_hid3 = tk.Radiobutton(model_without_optimization, text = '3',variable = num_hid, value = 3,command = enable_third)
num_hid3.grid(row = 0, column = 3)

ttk.Label(model_without_optimization, text='NUERONS IN FIRST LAYER').grid(row=1, column=0)
first_layer = tk.StringVar()
entry_first_layer = tk.Entry(model_without_optimization, textvariable=first_layer,state='disabled')
entry_first_layer.grid(row=1, column=1,pady=10)


ttk.Label(model_without_optimization, text='Activation Function').grid(row=1, column=2,padx=5)

comboFirstLayer = ttk.Combobox(model_without_optimization, 
                            values=[
                                    'relu', 
                                    'tanh',
                                    'sigmoid',
                                    'LeakyReLU'
                                    ])

comboFirstLayer.grid(column=3, row=1,pady=10)
comboFirstLayer.current(0)



ttk.Label(model_without_optimization, text='NUERONS IN SECOND LAYER').grid(row=2, column=0)
second_layer = tk.StringVar()
entry_second_layer = tk.Entry(model_without_optimization, textvariable=second_layer,state='disabled')
entry_second_layer.grid(row=2, column=1,pady=10)

ttk.Label(model_without_optimization, text='Activation Function').grid(row=2, column=2,padx=5)

comboSecondLayer = ttk.Combobox(model_without_optimization, 
                             values=[
                                    'relu', 
                                    'tanh',
                                    'sigmoid',
                                    'LeakyReLU'
                                    ])

comboSecondLayer.grid(column=3, row=2,pady=10)
comboSecondLayer.current(0)


ttk.Label(model_without_optimization, text='NUERONS IN THIRD LAYER').grid(row=3, column=0)
third_layer = tk.StringVar()
entry_third_layer = tk.Entry(model_without_optimization, textvariable=third_layer,state='disabled')
entry_third_layer.grid(row=3, column=1,pady=10)

ttk.Label(model_without_optimization, text='Activation Function').grid(row=3, column=2,padx=5)

comboThirdLayer = ttk.Combobox(model_without_optimization, 
                             values=[
                                    'relu', 
                                    'tanh',
                                    'sigmoid',
                                    'LeakyReLU'
                                    ])

comboThirdLayer.grid(column=3, row=3,pady=10)
comboThirdLayer.current(0)



### MODEL WITH OPTIMIZATION ######

model_with_optimization = ttk.Labelframe(model_frame, text = 'Model With Optimization')
model_with_optimization.grid(row=1,column=0)


radVar_neuron = tk.IntVar()

## One Hidden With Optimization ##


otoModelOne = tk.Radiobutton(model_with_optimization, text = 'One Hidden Layer',variable = num_hid, value = 4,command=enable_oto_one)
otoModelOne.grid(row = 0, column = 0)

ttk.Label(model_with_optimization, text='N_Min').grid(row=1, column=0)
n_min = tk.StringVar()
entry_n_min = tk.Entry(model_with_optimization, textvariable=n_min,state='disabled')
entry_n_min.grid(row=1, column=1,pady=10,padx=5)

ttk.Label(model_with_optimization, text='N_Max').grid(row=2, column=0)
n_max = tk.StringVar()
entry_n_max = tk.Entry(model_with_optimization, textvariable=n_max,state='disabled')
entry_n_max.grid(row=2, column=1,pady=10,padx=5)


## Two Hidden With Optimization ##

otoModelTwo = tk.Radiobutton(model_with_optimization, text = 'Two Hidden Layer',variable = num_hid, value = 5,command=enable_oto_two)
otoModelTwo.grid(row = 0, column = 2)

ttk.Label(model_with_optimization, text='N1_Min').grid(row=1, column=2)
n21_min = tk.StringVar()
entry_n21_min = tk.Entry(model_with_optimization, textvariable=n21_min,state='disabled')
entry_n21_min.grid(row=1, column=3,pady=10,padx=5)

ttk.Label(model_with_optimization, text='N1_Max').grid(row=2, column=2)
n21_max = tk.StringVar()
entry_n21_max = tk.Entry(model_with_optimization, textvariable=n21_max,state='disabled')
entry_n21_max.grid(row=2, column=3,pady=10,padx=5)

ttk.Label(model_with_optimization, text='N2_Min').grid(row=3, column=2)
n22_min = tk.StringVar()
entry_n22_min = tk.Entry(model_with_optimization, textvariable=n22_min,state='disabled')
entry_n22_min.grid(row=3, column=3,pady=10,padx=5)


ttk.Label(model_with_optimization, text='N2_Max').grid(row=4, column=2)
n22_max = tk.StringVar()
entry_n22_max = tk.Entry(model_with_optimization, textvariable=n22_max,state='disabled')
entry_n22_max.grid(row=4, column=3,pady=10,padx=5)



## Three Hidden With Optimization ##

otoModelThree = tk.Radiobutton(model_with_optimization, text = 'Three Hidden Layer',variable = num_hid, value = 6,command=enable_oto_three)
otoModelThree.grid(row = 0, column = 4)



ttk.Label(model_with_optimization, text='N1_Min').grid(row=1, column=4)
n31_min = tk.StringVar()
entry_n31_min = tk.Entry(model_with_optimization, textvariable=n31_min,state='disabled')
entry_n31_min.grid(row=1, column=5,pady=10,padx=5)

ttk.Label(model_with_optimization, text='N1_Max').grid(row=2, column=4)
n31_max = tk.StringVar()
entry_n31_max = tk.Entry(model_with_optimization, textvariable=n31_max,state='disabled')
entry_n31_max.grid(row=2, column=5,pady=10,padx=5)

ttk.Label(model_with_optimization, text='N2_Min').grid(row=3, column=4)
n32_min = tk.StringVar()
entry_n32_min = tk.Entry(model_with_optimization, textvariable=n32_min,state='disabled')
entry_n32_min.grid(row=3, column=5,pady=10,padx=5)

ttk.Label(model_with_optimization, text='N2_Max').grid(row=4, column=4)
n32_max = tk.StringVar()
entry_n32_max = tk.Entry(model_with_optimization, textvariable=n32_max,state='disabled')
entry_n32_max.grid(row=4, column=5,pady=10,padx=5)

ttk.Label(model_with_optimization, text='N3_Min').grid(row=5, column=4)
n33_min = tk.StringVar()
entry_n33_min = tk.Entry(model_with_optimization, textvariable=n33_min,state='disabled')
entry_n33_min.grid(row=5, column=5,pady=10,padx=5)

ttk.Label(model_with_optimization, text='N3_Max').grid(row=6, column=4)
n33_max = tk.StringVar()
entry_n33_max = tk.Entry(model_with_optimization, textvariable=n33_max,state='disabled')
entry_n33_max.grid(row=6, column=5,pady=10,padx=5)

################# HYPERPARAMETERS FRAME ##########################

hyper_frame = ttk.Labelframe(root,text = 'Hyperparameters')
hyper_frame.grid(row=3,column=1,columnspan=2,rowspan=2,sticky='w')

### Epoch ##

ttk.Label(hyper_frame, text='Epoch').grid(row=0, column=0,sticky='w')
n_epoch = tk.StringVar()
entry_n_epoch = tk.Entry(hyper_frame, textvariable=n_epoch).grid(row=0, column=1,sticky='w')


### Batch Size ###

ttk.Label(hyper_frame, text='Batch Size').grid(row=0, column=2,sticky='w')
n_batch = tk.StringVar()
entry_n_batch = tk.Entry(hyper_frame, textvariable=n_batch).grid(row=0, column=3,sticky='w')

## Optimizer##

ttk.Label(hyper_frame, text='Optimizer').grid(row=1, column=0,sticky='w')
comboOptimizer = ttk.Combobox(hyper_frame, 
                            values=[
                                    "Adam",                                  
                                    "Adagrad",
                                    "Adamax",
                                    "Adadelta",
                                    "SGD"])

comboOptimizer.grid(column=1, row=1,pady=10,sticky='w')
comboOptimizer.current(0)

## Loss Function ##

ttk.Label(hyper_frame, text='Loss Function').grid(row=1, column=2,sticky='w')
comboLoss = ttk.Combobox(hyper_frame, 
                            values=[
                                    "mean_squared_error", 
                                    "mean_absolute_error",
                                    "mape"])
comboLoss.grid(column=3, row=1,pady=10,sticky='w')
comboLoss.current(0)

### Learning Rate ##

ttk.Label(hyper_frame, text='Learning Rate').grid(row=2, column=0,sticky='w')
n_rate = tk.StringVar()
entry_n_rate = tk.Entry(hyper_frame, textvariable=n_rate).grid(row=2, column=1,sticky='w')

### Momentum ##

ttk.Label(hyper_frame, text='Momentum').grid(row=2, column=2,sticky='w')
n_momentum = tk.StringVar(value='0.0')
entry_n_momentum = tk.Entry(hyper_frame, textvariable=n_momentum).grid(row=2, column=3,sticky='w')


ttk.Label(hyper_frame, text='Model Type').grid(row=3, column=0)


radVar_modelType = tk.IntVar()

rad1_model = tk.Radiobutton(hyper_frame, text = 'MLP Model',variable = radVar_modelType, value = 1)
rad1_model.grid(row = 3, column = 1,sticky='w')

rad2_model = tk.Radiobutton(hyper_frame, text = 'LSTM Model',variable = radVar_modelType, value = 2)
rad2_model.grid(row = 3, column = 2,sticky='w')

rad3_model = tk.Radiobutton(hyper_frame, text = 'Bidirectional LSTM Model',variable = radVar_modelType, value = 3)
rad3_model.grid(row = 3, column = 3,sticky='w')


tk.Button(hyper_frame, text="Create Model",command=create_model).grid(row=4,column=0,pady=10,sticky='w')



ttk.Label(hyper_frame, text='Train Loss').grid(row=4,column=1,sticky='w')
trainError = tk.StringVar()
entry_trainError = tk.Entry(hyper_frame, textvariable=trainError).grid(row=4, column=2,sticky='w')


tk.Button(hyper_frame, text = 'Save Model', command=model_save).grid(row=4,column=3,pady=10,sticky='w')


ttk.Label(hyper_frame, text='Best Model Neuron Numbers').grid(row=5,column=0,sticky='w')
firstLayer = tk.StringVar()
entry_firstLayer = tk.Entry(hyper_frame, textvariable=firstLayer, width = 5, state = 'disabled').grid(row=5, column=1,sticky='w')


secondLayer = tk.StringVar()
entry_secondLayer = tk.Entry(hyper_frame, textvariable=secondLayer, width = 5, state = 'disabled').grid(row=5, column=2,sticky='w')


thirdLayer = tk.StringVar()
entry_thirdLayer = tk.Entry(hyper_frame, textvariable=thirdLayer, width = 5, state = 'disabled').grid(row=5, column=3,sticky='w')






#################   TEST FRAME ###################################

test_frame = ttk.Labelframe(root,text = 'Test Model')
test_frame.grid(row=3,column=3,rowspan=2,padx=5,sticky='w')



tk.Label(test_frame, text='Test File Path').grid(row=1, column=0,sticky='W')
v_test = tk.StringVar()
entry_test = tk.Entry(test_frame, textvariable=v_test)
entry_test.grid(row=1, column=1,sticky='w')
test_button = tk.Button(test_frame, text='Get Test Set', command=import_test_data)
test_button.grid(row=1, column=2,sticky='w')




ttk.Label(test_frame, text='# of Forecast').grid(row=0, column=0,pady = 5,sticky='w')
number_forecast = tk.StringVar()
#radVar_test = tk.IntVar()
#rad1 = tk.Radiobutton(test_frame, text = 'As Percent',variable = radVar_test, value = 1).grid(row = 1, column = 1)
#rad2 = tk.Radiobutton(test_frame, text = 'As Number',variable = radVar_test, value = 2).grid(row = 1, column = 2)
entry_number_forecast = tk.Entry(test_frame, textvariable=number_forecast).grid(row=0, column=1, pady=5,sticky='w')

forecast_button = tk.Button(test_frame, text='Values', command=show_forecasted_vals)
forecast_button.grid(row=0, column=2,sticky='w')




radVar_model = tk.IntVar()
rad1_model = tk.Radiobutton(test_frame, text = 'Use Created Model',variable = radVar_model, value = 1,command=disable_model_browser).grid(row = 3, column = 0,sticky='w')
rad2_model = tk.Radiobutton(test_frame, text = 'Select Model',variable = radVar_model, value = 2,command=enable_model_browser).grid(row = 3, column = 1,sticky='w')



tk.Label(test_frame, text='Model File Path').grid(row=4, column=0,sticky='w')
v_model = tk.StringVar()
entry_model = tk.Entry(test_frame, textvariable=v_model,state ='disabled')
entry_model.grid(row=4, column=1,sticky='w')
model_button = tk.Button(test_frame, text='Get Model', command=import_model,state ='disabled')
model_button.grid(row=4, column=2,sticky='w')



tk.Button(test_frame, text='Test Model', command=test_model).grid(row=5, column=0,sticky='w')

tk.Button(test_frame, text='Actual vs Forecasted Graph', command=showActualvsForecasted).grid(row=5, column=1,sticky='w')



ttk.Label(test_frame, text='Test MAPE').grid(row=6, column=0,sticky='w')
test_mape = tk.StringVar()
entry_test_mape = tk.Entry(test_frame, textvariable=test_mape).grid(row=6, column=1,sticky='w')



root.mainloop()















