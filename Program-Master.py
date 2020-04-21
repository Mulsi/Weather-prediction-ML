import pandas as pd
import numpy as np
import datetime
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

data = pd.read_excel('C:/Users/mulej/Desktop/Petrol/Weather-prediction-ML/Excel-data/Project-ML-data.xlsx')
df = pd.DataFrame(data, columns= ['all_datetimes', 'wind_dir', 'wind_speed'])
# df['all_datetimes'] = pd.to_datetime(df['all_datetimes'],format= '%Y-%m-%d')
df['all_datetimes'] = pd.to_datetime(df['all_datetimes'], infer_datetime_format=True, errors='coerce')
Every_date = df.all_datetimes
# print(Every_date)
# Every_date = datetime.datetime.strptime(str(Every_date), '%Y-%m-%d')
# DataTypes = df.dtypes
# print(DataTypes)
# summary = data.describe()
# print(summary)


def train_data():
    ## Train & test data
    X = data.drop(["all_datetimes"], axis=1)
    # X = X.drop(["wind_dir"], axis=1)
    y = X.wind_speed

    # new_dates = []
    # counter = 0
    # dates = X.Every_date
    # for Every_date in dates:
    #     Every_date = datetime.datetime.strptime(Every_date, '%Y-%m-%d')
    # X.Every_date = new_dates    



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    print(X_train.shape)
    print(X_test.shape)


    ## Standardizacija
    scaler = preprocessing.StandardScaler().fit(X_train)
    train_scaled = scaler.transform(X_train)
    # print(train_scaled)
    test_scaled = scaler.transform(X_test)
    # print(test_scaled)


    ## Declaring data preprocessing steps
    pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))
    ## Declare hyperparameters to tune
    hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 
        'randomforestregressor__max_depth': [None, 5, 3, 1], }


    ## Tune model using cross-validation pipeline
    clf = GridSearchCV(pipeline, hyperparameters, cv=10)
    # clf = LinearRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    # print(pred)
    score = r2_score(y_test, pred)
    meanSquaredError = mean_squared_error(y_test, pred)
    print(score)
    print('The mean squared error: %.2f' % mean_squared_error(y_test, pred))


    ## Model se shrani za kasnejšo uporabo
    joblib.dump(clf, 'weather_predictor.pkl')



## Ta del je samo za vpogled kakšno je bilo vreme v preteklosti. V bistvu sploh ni uporabljen
def get_the_weather(dates):
    wind = data.wind_speed
    all_dates = data.all_datetimes
    # all_dates = str(all_dates)
    # if (all_dates == dates):
    #     print("Pali")
    #     return wind

    # for i in range(0, len(all_dates)):
    #     datetime_object = datetime.datetime.strptime(all_dates[0], '0')
    #     datetime_object = datetime.datetime.strptime(all_dates[1:], '%y-%d-%m')
    #     datetime_object = datetime.datetime.fromtimestamp(all_dates)
    #     print("Success.")
    #     if (datetime_object == date):
    #         return wind[i]



    # for i in range(0, len(all_dates)):
        # all_dates = str(all_dates)
        # print(all_dates)
        # print(type(all_dates))

        # datetime_object = datetime.datetime.strptime(all_dates[0], '0')
        # print("Success.")
        # print(datetime_object)
        # print(all_dates)
        # datetime_object = datetime.datetime.strptime(all_dates[1], '')
        # datetime_object = datetime.datetime.strptime(all_dates[1:], '%y-%d-%m') #(all_dates[1:], '%Y-%m-%d')
        # datetime_object = datetime.datetime.strptime(all_dates[1:], '%y-%m-%d %H:%M:%S')
        # print("Success on the second one.")

        # datetime_object = datetime.datetime.strptime(all_dates, '%y-%m-%d %H:%M:%S')
        # print("Do tuki prideš.")
        # return wind[i]
        # if (datetime_object == date):
            # return wind[i]


def predict_weather():
    clf = joblib.load('weather_predictor.pkl')
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
    print(day)
    date = (day - datetime.datetime(1970,1,1)).total_seconds()
    

    day_x = str(year) + '-' + str(month) + '-' +str(specific_day)
    day_x = datetime.datetime.strptime(day_x, '%Y-%m-%d')
    print(day_x)
    date_x = (day_x - datetime.datetime(1970,1,1)).total_seconds()

    X = [[date, date_x]]
    # X = [date]
    print('\n')
    print('-' * 48)
    print('The wind speed is predicted to be: ' + str(clf.predict(X)[0])) #Včasih je bilo str(clf.predict(X)[0])
    # print('The wind speed was actually: ' + str(get_the_weather(day)))
    # print('-', * 48)
    print('\n')


def run_menu():
    print('*' * 48)
    print('-' *10 + ' What would you like to do? ' + '-' *10)
    print("\n")
    print('1. Look up the weather on a specific day')
    print('2. Predict the weather on a specific day')
    print('\n')

    option = input('Enter option: ')
    # print(option)

    # while True:
    #     if option == 2 or option == 1 or option == 9:
    #         print('Ojla živ.')
    #         break
    #     option = input('Enter option: ')
    #     print('Dela.')
    return option


def run_program(option):
    predict_weather()
    # if option == 1:
    #     print('1')
    # elif option == 2:
    #     predict_weather()

if __name__ == '__main__':
    train_data()
    # print('Pali.')

    while True:
        option = run_menu()
        if option == 9:
            break
        else:
            run_program(option)    

    



