import pandas as pd
import numpy as np
import datetime
import sys
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
from pylab import * 



data = pd.read_csv('C:/Users/mulej/Desktop/Petrol/Weather-prediction-ML/Excel-data/Project-ML-data-old.csv', names=['all_datetimes', 'wind_dir', 'wind_speed'], sep=';')
print(data)
# described = data.describe()
# print(described)


## Train & test data
# Y je tista spremenljivka ki jo iščemo pri predikciji, X so vse spremenljivke ki jih uporabimo za sestavitev modela.
def train_data():
    y = data.wind_speed
    X = data.drop(["wind_speed"], axis=1)
    
    
    
    # plt.scatter(data.all_datetimes, data.wind_speed)
    # plt.show()

    ## Convertion da tip datuma paše v model
    new_dates = []
    dates = X.all_datetimes
    for all_datetimes in dates:
        all_datetimes = datetime.datetime.strptime(all_datetimes, '%Y-%m-%d')
        all_datetimes2 = (all_datetimes - datetime.datetime(1970,1,1)).total_seconds()
        new_dates.append(all_datetimes2)
    X.all_datetimes = new_dates


    ## Train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    print(X_train.shape)
    print(X_test.shape)

    # Reshaped = np.dot(data.wind_dir,data.power)
    # print(Reshaped)





    ## Standardizacija
    # Proces:
    # 1. Fit the transformer on the training set (saving the means and standard deviations)
    # 2. Apply the transformer to the training set (scaling the training data)
    # 3. Apply the transformer to the test set (using the same means and standard deviations)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    # print(X_train_scaled)

    test_scaler = preprocessing.StandardScaler().fit(X_test)
    X_test_scaled = scaler.transform(X_test)


    ## Declaring data preprocessing steps
    # a modeling pipeline that first transforms the data using StandardScaler() 
    # and then fits a model using a random forest regressor.
    pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))
    ## Declare hyperparameters to tune
    hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 
        'randomforestregressor__max_depth': [None, 5, 3, 1], }


    ## Tune model using cross-validation pipeline (Več opcij kateri model vzameš)
    clf = GridSearchCV(pipeline, hyperparameters, cv=10)
    # clf = LinearRegression()
    # clf = tree.DecisionTreeClassifier()
    print(X_train.shape)
    print(y_train.shape)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    TestScore = r2_score(y_test, y_pred) ## Score regresije. Blizu 1: dober rezultat, bližje 0: slab rezultat. 
    meanSquaredError = mean_squared_error(y_test, y_pred)
    print('The test score is: %.2f' % TestScore)
    print('The mean squared error is: %.2f' % mean_squared_error(y_test, y_pred))
    

    ## Model se shrani za kasnejšo uporabo
    joblib.dump(clf, 'weather_predictor-old.pkl')



## Ta del je samo za vpogled kakšno je bilo vreme v preteklosti. Trenutno sploh ni uporabljen
def get_the_weather(dates):
    wind = data.wind_speed
    all_dates = data.all_datetimes



def predict_weather():
    clf = joblib.load('weather_predictor-old.pkl')
    print('Enter a date you would like to predict')
    print('\n')
    option = input('Year: ')
    year = option
    option = input('Month: ')
    month = option
    option = input('Day: ')
    specific_day = option
    

    day = str(year) + '-' + str(month) + '-' +str(specific_day)
    day = datetime.datetime.strptime(day, '%Y-%m-%d')
    # print(day)
    date = (day - datetime.datetime(1970,1,1)).total_seconds()
    

    day_x = str(year) + '-' + str(month) + '-' +str(specific_day)
    day_x = datetime.datetime.strptime(day_x, '%Y-%m-%d')
    # print(day_x)
    date_x = (day_x - datetime.datetime(1970,1,1)).total_seconds()

    X = [[date, date_x]]
    print('\n')
    print('-' * 48)
    print('The wind speed is predicted to be: ' + str(clf.predict(X)[0]))
    # print('The wind speed was actually: ' + str(get_the_weather(day)))
    # print('-', * 48)
    print('\n')


def run_menu():
    print('*' * 48)
    print('-' *10 + ' What would you like to do? ' + '-' *10)
    print("\n")
    print('1. Predict wind speed on a specific day')
    # print('2. Exit')
    print('\n')

    option = input('Enter option: ')
    print('\n')

    return option




def run_program(option):
    if option == '1':
        predict_weather()
    else:
        print('Input 1 for prediction.')
        print('\n')
        
        


if __name__ == '__main__':
    train_data()

    while True:
        option = run_menu()
        if option == 9:
            break
        else:
            run_program(option)    

    



