#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def regresiData (X,Y,Z):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np 
    import sklearn
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    import math
    from math import sqrt
    data = pd.read_csv(Z, usecols=[X,Y] )
    x = data[X].values[:,np.newaxis]
    y=  data[Y].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
    regresi = LinearRegression()
    regresi.fit(x_train, y_train)
    y_test_predict =regresi.predict(x_test)
    print('VISUALISASI TRAINING DATA DAN VISUALISASI TEST DATA')
    #visulisasi training proses
    plt.figure(figsize=(10,8))
    #biru adalah data observasi
    plt.scatter(x_train, y_train, color='blue')
    #garis merah adalah prediksi dari mesin learning
    plt.plot(x_train, regresi.predict(x_train), color='red')
    plt.title('PENGARUH {0} TERHADAP {1}'.format(X,Y))
    plt.xlabel(X)
    plt.ylabel(Y)
    #visulisasi training proses
    print ('*'*100)
    print ('*'*100)
    plt.figure(figsize=(10,8))
    #biru adalah data observasi
    plt.scatter(x_test, y_test, color='blue')
    #garis merah adalah prediksi dari mesin learning
    plt.plot(x_test, regresi.predict(x_test), color='red')
    plt.title('PENGARUH {0} TERHADAP {1}'.format(X,Y))
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.show()
    y_test_predict =regresi.predict(x_test)
    #rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
    mse = sklearn.metrics.mean_squared_error(y_test, y_test_predict)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_test_predict)
    print ('*'*100)
    print ('*'*100)
    beda_mean = y_test.mean() - y_test_predict.mean()
    print('perbedaan mean dari tes dan prediski adalah,' ,beda_mean)
    print('Hasil Error RMSE',rmse)
    print('HASIL ERROR R2',r2)
    corelasi = data.corr(method ='pearson')
    print ('DATA KORELASI ANTARA {0} DENGAN {1} :' .format(X,Y), corelasi.iloc[1][0])
    
    

regresiData ("GDP per capita","Score",'happy_country.csv')

