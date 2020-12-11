# Air-Travelers
ML modeling for number of air travelers in the United States
Travel in 2020
The year 2020 has been filled with many different events that have impacted various commercial areas. Specifically, the travel industry in the United States has seen a significant down tick in the number of people traveling. The purpose of this project is to collect data from various different sources to see if we can predict with high accuracy the number of people that would travel on a single day in 2020, specifically flying. We will be using data for number of travelers from the TSA website, data from the ourworldindata.org for covid case counts per day, stock market data from yahoo finance, and various other sources to know dates of travel bans. We want to see if the economic status, public health status, sports status, or CDC guidelines can be used to predict with high accuracy the number of air travelers in the United States.

Import Libraries
#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import math
from bs4 import BeautifulSoup
%matplotlib inline
from dfply import *
import matplotlib.dates as mdates
Read in data
#Read in covid case and death count data
covid = pd.read_csv('daily-covid-cases-deaths.csv')
#Check to see if read in properly
covid.head()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
Entity	Code	Date	Daily new confirmed cases of COVID-19	Daily new confirmed deaths due to COVID-19
0	Afghanistan	AFG	2020-01-23	0	0
1	Afghanistan	AFG	2020-01-24	0	0
2	Afghanistan	AFG	2020-01-25	0	0
3	Afghanistan	AFG	2020-01-26	0	0
4	Afghanistan	AFG	2020-01-27	0	0
#Read in stock market data
stock = pd.read_csv('^DJI.csv')
#Check to see if read in properly
stock.head()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
Date	Open	High	Low	Close	Adj Close	Volume
0	2019-12-10	27900.650391	27949.019531	27804.000000	27881.720703	27881.720703	213250000
1	2019-12-11	27867.310547	27925.500000	27801.800781	27911.300781	27911.300781	213510000
2	2019-12-12	27898.339844	28224.949219	27859.869141	28132.050781	28132.050781	277740000
3	2019-12-13	28123.640625	28290.730469	28028.320313	28135.380859	28135.380859	250660000
4	2019-12-16	28191.669922	28337.490234	28191.669922	28235.890625	28235.890625	286770000
#Read in data for number of travelers
tsa_data = pd.read_csv('tsa_data.csv')
#Check to see if read in properly
tsa_data.head()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
Date	Total Traveler Throughput	Total Traveler Throughput (1 Year Ago - Same Weekday)
0	12/9/2020	564,372	2,020,488
1	12/8/2020	501,513	1,897,051
2	12/7/2020	703,546	2,226,290
3	12/6/2020	837,137	2,292,079
4	12/5/2020	629,430	1,755,801
#Sources for NBA and other sport suspensions
# https://www.nba.com/news/nba-suspend-season-following-wednesdays-games
# https://bleacherreport.com/articles/2880569-timeline-of-coronavirus-impact-on-sports
#Create dataset to show when first major sports league suspended season (NBA) and when it restarted
nba = pd.DataFrame()

sports_maybe = []

for i in range(1, 366):
    if i < 71:
        sports_maybe.append(1)
    elif i >= 71 and i < 189:
        sports_maybe.append(0)
    else:
        sports_maybe.append(1)
        
nba['Games'] = sports_maybe
        
nba.head()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
Games
0	1
1	1
2	1
3	1
4	1
#Source for first mask recommendation made by CDC
# https://www.npr.org/sections/coronavirus-live-updates/2020/04/03/826219824/president-trump-says-cdc-now-recommends-americans-wear-cloth-masks-in-public
#Create dataset to show when masks were recommended

masks = pd.DataFrame()

masks_lst = []

for i in range(1, 366):
    if i < 93:
        masks_lst.append(0)
    else:
        masks_lst.append(1)

masks['recommendation'] = masks_lst
#Check dataset
masks
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
recommendation
0	0
1	0
2	0
3	0
4	0
...	...
360	1
361	1
362	1
363	1
364	1
365 rows × 1 columns

Clean up data and Create main dataframe
#Clean up covid dataset to only include United States
usa_covid = covid.loc[covid['Entity'] == 'United States']
#Check earliest date for tsa dates
tsa_data.tail()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
Date	Total Traveler Throughput	Total Traveler Throughput (1 Year Ago - Same Weekday)
279	3/5/2020	2,130,015	2,402,692
280	3/4/2020	1,877,401	2,143,619
281	3/3/2020	1,736,393	1,979,558
282	3/2/2020	2,089,641	2,257,920
283	3/1/2020	2,280,522	2,301,439
The earliest date for the flights begin on March 1, 2020. We will need to drop all other observations before that date.

#Drop all dates from stock before March 1
stock = stock[(stock['Date'] >= '2020-03-01') & (stock['Date'] <= '2020-12-09')].reset_index()
#Drop all dates from covid before March 1
usa_covid = usa_covid[(usa_covid['Date'] >= '2020-03-01') & (usa_covid['Date'] <= '2020-12-09')].reset_index()
#Check to see that all worked out
stock.head()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
index	Date	Open	High	Low	Close	Adj Close	Volume
0	55	2020-03-02	25590.509766	26706.169922	25391.960938	26703.320313	26703.320313	637200000
1	56	2020-03-03	26762.470703	27084.589844	25706.279297	25917.410156	25917.410156	647080000
2	57	2020-03-04	26383.679688	27102.339844	26286.310547	27090.859375	27090.859375	457590000
3	58	2020-03-05	26671.919922	26671.919922	25943.330078	26121.279297	26121.279297	477370000
4	59	2020-03-06	25457.210938	25994.380859	25226.619141	25864.779297	25864.779297	599780000
usa_covid.head()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
index	Entity	Code	Date	Daily new confirmed cases of COVID-19	Daily new confirmed deaths due to COVID-19
0	61551	United States	USA	2020-03-01	7	0
1	61552	United States	USA	2020-03-02	23	5
2	61553	United States	USA	2020-03-03	19	1
3	61554	United States	USA	2020-03-04	33	4
4	61555	United States	USA	2020-03-05	77	1
#Drop the index columns
stock.drop(labels = 'index', axis = 1, inplace = True)
usa_covid.drop(labels = 'index', axis = 1, inplace = True)
#Check to see it worked
stock.head()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
Date	Open	High	Low	Close	Adj Close	Volume
0	2020-03-02	25590.509766	26706.169922	25391.960938	26703.320313	26703.320313	637200000
1	2020-03-03	26762.470703	27084.589844	25706.279297	25917.410156	25917.410156	647080000
2	2020-03-04	26383.679688	27102.339844	26286.310547	27090.859375	27090.859375	457590000
3	2020-03-05	26671.919922	26671.919922	25943.330078	26121.279297	26121.279297	477370000
4	2020-03-06	25457.210938	25994.380859	25226.619141	25864.779297	25864.779297	599780000
usa_covid.head()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
Entity	Code	Date	Daily new confirmed cases of COVID-19	Daily new confirmed deaths due to COVID-19
0	United States	USA	2020-03-01	7	0
1	United States	USA	2020-03-02	23	5
2	United States	USA	2020-03-03	19	1
3	United States	USA	2020-03-04	33	4
4	United States	USA	2020-03-05	77	1
len(usa_covid)
284
len(stock)
198
#Attempt to merge covid and stock
main_df = usa_covid.merge(stock, how = 'left', left_on = 'Date', right_on = 'Date')
#Change the NBA dataframe and Mask dataframe to match the dates

nba = nba.iloc[60 : 344]
masks = masks.iloc[60 : 344]
nba = nba.reset_index(drop = True)
masks = masks.reset_index(drop = True)
#Add masks and nba to the main dataframe
main_df['Sports'] = nba['Games']
main_df['Masks'] = masks['recommendation']
#Check to see if it's all there
main_df.head()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
Entity	Code	Date	Daily new confirmed cases of COVID-19	Daily new confirmed deaths due to COVID-19	Open	High	Low	Close	Adj Close	Volume	Sports	Masks
0	United States	USA	2020-03-01	7	0	NaN	NaN	NaN	NaN	NaN	NaN	1	0
1	United States	USA	2020-03-02	23	5	25590.509766	26706.169922	25391.960938	26703.320313	26703.320313	637200000.0	1	0
2	United States	USA	2020-03-03	19	1	26762.470703	27084.589844	25706.279297	25917.410156	25917.410156	647080000.0	1	0
3	United States	USA	2020-03-04	33	4	26383.679688	27102.339844	26286.310547	27090.859375	27090.859375	457590000.0	1	0
4	United States	USA	2020-03-05	77	1	26671.919922	26671.919922	25943.330078	26121.279297	26121.279297	477370000.0	1	0
#Drop entity and code
main_df.drop(labels = ['Entity', 'Code'], axis = 1, inplace = True)
main_df.rename({'Daily new confirmed cases of COVID-19' : 'Cases', 'Daily new confirmed deaths due to COVID-19' : 'Deaths'}, axis = 1, inplace = True)
#Replace NANs 
main_df.fillna({'Open' : main_df['Open'].median(), 
               'High' : main_df['High'].median(),
               'Low' : main_df['Low'].median(), 
               'Close' : main_df['Close'].median(),
               'Adj Close' : main_df['Adj Close'].median(),
               'Volume' : main_df['Volume'].median()}, inplace = True)
#Check it out
main_df.isna().sum()
Date         0
Cases        0
Deaths       0
Open         0
High         0
Low          0
Close        0
Adj Close    0
Volume       0
Sports       0
Masks        0
dtype: int64
#Finally add the tsa dataframe
main_df['Travelers 2020'] = tsa_data['Total Traveler Throughput']
main_df['Travelers 2019'] = tsa_data['Total Traveler Throughput (1 Year Ago - Same Weekday)']
#Look at final product?
main_df
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
Date	Cases	Deaths	Open	High	Low	Close	Adj Close	Volume	Sports	Masks	Travelers 2020	Travelers 2019
0	2020-03-01	7	0	26668.264649	26855.940429	26500.794922	26668.174805	26668.174805	401285000.0	1	0	564,372	2,020,488
1	2020-03-02	23	5	25590.509766	26706.169922	25391.960938	26703.320313	26703.320313	637200000.0	1	0	501,513	1,897,051
2	2020-03-03	19	1	26762.470703	27084.589844	25706.279297	25917.410156	25917.410156	647080000.0	1	0	703,546	2,226,290
3	2020-03-04	33	4	26383.679688	27102.339844	26286.310547	27090.859375	27090.859375	457590000.0	1	0	837,137	2,292,079
4	2020-03-05	77	1	26671.919922	26671.919922	25943.330078	26121.279297	26121.279297	477370000.0	1	0	629,430	1,755,801
...	...	...	...	...	...	...	...	...	...	...	...	...	...
279	2020-12-05	213881	2254	26668.264649	26855.940429	26500.794922	26668.174805	26668.174805	401285000.0	1	1	2,130,015	2,402,692
280	2020-12-06	175664	1113	26668.264649	26855.940429	26500.794922	26668.174805	26668.174805	401285000.0	1	1	1,877,401	2,143,619
281	2020-12-07	192435	1404	30233.029297	30233.029297	29967.220703	30069.789063	30069.789063	365810000.0	1	1	1,736,393	1,979,558
282	2020-12-08	215878	2546	29997.949219	30246.220703	29972.070313	30173.880859	30173.880859	311190000.0	1	1	2,089,641	2,257,920
283	2020-12-09	221267	3124	30229.810547	30319.699219	29951.849609	30068.810547	30068.810547	380520000.0	1	1	2,280,522	2,301,439
284 rows × 13 columns

Our dataframe is all prepared and ready to do some EDA.

EDA
#Look at summary statistics
main_df.describe()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
Cases	Deaths	Open	High	Low	Close	Adj Close	Volume	Sports	Masks
count	284.000000	284.000000	284.000000	284.000000	284.000000	284.000000	284.000000	2.840000e+02	284.000000	284.000000
mean	54177.947183	1018.915493	26390.153733	26619.178649	26157.870468	26392.210869	26392.210869	4.298291e+08	0.584507	0.887324
std	48470.687826	649.333891	2063.128829	1986.881384	2152.148505	2064.710643	2064.710643	1.117051e+08	0.493677	0.316755
min	7.000000	0.000000	19028.359375	19121.009766	18213.650391	18591.929688	18591.929688	1.770400e+08	0.000000	0.000000
25%	25213.500000	537.250000	25900.005859	26102.789551	25609.060058	25855.424317	25855.424317	3.733825e+08	0.000000	1.000000
50%	40737.000000	940.500000	26668.264649	26855.940429	26500.794922	26668.174805	26668.174805	4.012850e+08	1.000000	1.000000
75%	61185.250000	1295.500000	27542.353027	27810.099610	27447.347656	27602.400391	27602.400391	4.402475e+08	1.000000	1.000000
max	227828.000000	3124.000000	30233.029297	30319.699219	29989.560547	30218.259766	30218.259766	9.082600e+08	1.000000	1.000000
#Change Travelers data type
for i in range(0, len(main_df)):
    main_df['Travelers 2020'][i] = main_df['Travelers 2020'][i].replace(',', '')
    main_df['Travelers 2019'][i] = main_df['Travelers 2019'][i].replace(',', '')
C:\Users\thebs\anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  This is separate from the ipykernel package so we can avoid doing imports until
C:\Users\thebs\anaconda3\lib\site-packages\ipykernel_launcher.py:4: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  after removing the cwd from sys.path.
main_df['Travelers 2020'] = pd.to_numeric(main_df['Travelers 2020'])
main_df['Travelers 2019'] = pd.to_numeric(main_df['Travelers 2019'])
#All summary statistics
main_df.describe()
<style scoped> .dataframe tbody tr th:only-of-type { vertical-align: middle; }
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
Cases	Deaths	Open	High	Low	Close	Adj Close	Volume	Sports	Masks	Travelers 2020	Travelers 2019
count	284.000000	284.000000	284.000000	284.000000	284.000000	284.000000	284.000000	2.840000e+02	284.000000	284.000000	2.840000e+02	2.840000e+02
mean	54177.947183	1018.915493	26390.153733	26619.178649	26157.870468	26392.210869	26392.210869	4.298291e+08	0.584507	0.887324	6.383851e+05	2.375686e+06
std	48470.687826	649.333891	2063.128829	1986.881384	2152.148505	2064.710643	2064.710643	1.117051e+08	0.493677	0.316755	4.012484e+05	2.461785e+05
min	7.000000	0.000000	19028.359375	19121.009766	18213.650391	18591.929688	18591.929688	1.770400e+08	0.000000	0.000000	8.753400e+04	1.591158e+06
25%	25213.500000	537.250000	25900.005859	26102.789551	25609.060058	25855.424317	25855.424317	3.733825e+08	0.000000	1.000000	3.466970e+05	2.193565e+06
50%	40737.000000	940.500000	26668.264649	26855.940429	26500.794922	26668.174805	26668.174805	4.012850e+08	1.000000	1.000000	6.339280e+05	2.430741e+06
75%	61185.250000	1295.500000	27542.353027	27810.099610	27447.347656	27602.400391	27602.400391	4.402475e+08	1.000000	1.000000	7.982395e+05	2.556494e+06
max	227828.000000	3124.000000	30233.029297	30319.699219	29989.560547	30218.259766	30218.259766	9.082600e+08	1.000000	1.000000	2.280522e+06	2.882915e+06
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
#Graph travel numbers from this year and last year
sns.jointplot(x = 'Travelers 2020', y = 'Travelers 2019', data = main_df)
<seaborn.axisgrid.JointGrid at 0x177d2a53b88>
png

As we can clearly see, the numbers for 2020 have been significantly smaller than from 2019

#Graph travelers 2020 vs cases
sns.lmplot(x = 'Travelers 2020', y = 'Cases', data = main_df)
<seaborn.axisgrid.FacetGrid at 0x177d40e7c08>
png

Very interesting shape to this...

#Plot line graph of number of cases
sns.relplot(x = 'Date', y = 'Cases', kind = 'line', data = main_df)
<seaborn.axisgrid.FacetGrid at 0x177d0f100c8>
png

#Plot count of days with masks vs no masks
sns.countplot(x = 'Masks', data = main_df)
<matplotlib.axes._subplots.AxesSubplot at 0x177d2123988>
png

#Plot number of travelers vs number of cases
plt.figure(figsize=(10,6))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.plot('Date', 'Travelers 2020', data = main_df, label = 'Travelers')
plt.plot('Date', 'Cases', data = main_df, label = 'Cases')

plt.legend()
<matplotlib.legend.Legend at 0x177d901ac88>
png

Models
We will now fit our data to different machine learning models. We will be fitting the data to a linear, KNN, ridge, and decision tree model. We will run MAE, MSE, RMSE, and CV sc

Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
#Separate response from explanatory
y = main_df['Travelers 2020']

X = main_df[['Cases', 'Deaths', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sports', 'Masks', 'Travelers 2019']]
#Separate training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()
#Fit a linear regression model
lm.fit(X_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
#Show coefficients
print('Coefficients: \n', lm.coef_)
Coefficients: 
 [ 4.34770099e+00  1.83464093e+02  1.17004012e+02 -1.82928386e+02
  6.75991449e+01 -3.32849264e+00 -3.32849263e+00  5.09455846e-05
 -3.70770896e+05 -5.31593389e+05  2.44579269e-01]
#Gather predictions from fitted model
predictions = lm.predict(X_test)
#Plot predictions vs actual
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
Text(0, 0.5, 'Predicted Y')
png

#Generate metrics
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('R^2: ', metrics.r2_score(y_test, predictions))
MAE: 209582.34029446437
MSE: 89312463686.86139
RMSE: 298851.9092909754
R^2:  0.5075655165151798
#CV score
print('Cross validation score: ',-1 * ((cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')).mean()))
Cross validation score:  279372589873.1498
With a RMSE of 298851 and CV score of 279372589873.1498, I would say that this model isn't the best for predicting the number of airplane travelers in 2020, but 2020 has been hard for anyone to predict so it's not the worst thing to come from 2020.

#Plot residuals
sns.distplot((y_test-predictions),bins=60);
png

At least the residuals look normal? That's good? Our data really isn't that linear though, so linear regression assumptions are iffy on this data

np.std(main_df['Travelers 2020'])
400541.3894473091
KNN
from sklearn.neighbors import KNeighborsRegressor
#KNN with 5
knn = KNeighborsRegressor(n_neighbors=5)
#Fit the model
knn.fit(X_train, y_train)
KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                    metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                    weights='uniform')
#Make predictions
knn_preds = knn.predict(X_test)
#Get metrics
print('MAE:', metrics.mean_absolute_error(y_test, knn_preds))
print('MSE:', metrics.mean_squared_error(y_test, knn_preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, knn_preds)))
MAE: 317331.5046511628
MSE: 198213011157.18558
RMSE: 445211.19837351976
#Get CV score
print('Cross validation score: ',-1 * ((cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error')).mean()))
Cross validation score:  230145124472.75333
Well, this model did not do any better at predicting with accuracy the amount of travelers flying on a given day in 2020 since the RMSE is much higher than that of the lm model. Let's try to find a proper amount of K with GridSearch.

from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors':list(range(1,30))}
knn = KNeighborsRegressor()
clf = GridSearchCV(knn, parameters)
clf.fit(X, y)
GridSearchCV(cv=None, error_score=nan,
             estimator=KNeighborsRegressor(algorithm='auto', leaf_size=30,
                                           metric='minkowski',
                                           metric_params=None, n_jobs=None,
                                           n_neighbors=5, p=2,
                                           weights='uniform'),
             iid='deprecated', n_jobs=None,
             param_grid={'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                         23, 24, 25, 26, 27, 28, 29]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
clf.best_params_
{'n_neighbors': 28}
knn = KNeighborsRegressor(n_neighbors=28)
knn.fit(X_train, y_train)
KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                    metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                    weights='uniform')
knn_preds = knn.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, knn_preds))
print('MSE:', metrics.mean_squared_error(y_test, knn_preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, knn_preds)))
MAE: 317331.5046511628
MSE: 198213011157.18558
RMSE: 445211.19837351976
Looks like we get the same results. KNN is not very good at and not as good as linear regression.

Ridge
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train, y_train)
C:\Users\thebs\anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=2.12647e-18): result may not be accurate.
  overwrite_a=True).T





Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
ridge_preds = ridge.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, ridge_preds))
print('MSE:', metrics.mean_squared_error(y_test, ridge_preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, ridge_preds)))
print('R^2: ', metrics.r2_score(y_test, ridge_preds))
MAE: 211176.6644662734
MSE: 90417914812.4394
RMSE: 300695.7179815493
R^2:  0.5014704853003841
#CV score
print('Cross validation score: ',-1 * ((cross_val_score(ridge, X, y, cv=10, scoring='neg_mean_squared_error')).mean()))
Cross validation score:  289037063937.7112


C:\Users\thebs\anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=2.45773e-18): result may not be accurate.
  overwrite_a=True).T
C:\Users\thebs\anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.48952e-18): result may not be accurate.
  overwrite_a=True).T
C:\Users\thebs\anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.46358e-18): result may not be accurate.
  overwrite_a=True).T
C:\Users\thebs\anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.48107e-18): result may not be accurate.
  overwrite_a=True).T
C:\Users\thebs\anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.48031e-18): result may not be accurate.
  overwrite_a=True).T
C:\Users\thebs\anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.49566e-18): result may not be accurate.
  overwrite_a=True).T
C:\Users\thebs\anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.5159e-18): result may not be accurate.
  overwrite_a=True).T
C:\Users\thebs\anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.49879e-18): result may not be accurate.
  overwrite_a=True).T
C:\Users\thebs\anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.49846e-18): result may not be accurate.
  overwrite_a=True).T
C:\Users\thebs\anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.49514e-18): result may not be accurate.
  overwrite_a=True).T
Comparing this model's RMSE and CV score to that of Linear regression and KNN, it is the middle man. It performed slightly worse than the Linear regression model and better than the KNN model.

Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
decTree = DecisionTreeRegressor()
decTree.fit(X_train, y_train)
DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=None,
                      max_features=None, max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, presort='deprecated',
                      random_state=None, splitter='best')
dec_preds = decTree.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, dec_preds))
print('MSE:', metrics.mean_squared_error(y_test, dec_preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dec_preds)))
print('R^2: ', metrics.r2_score(y_test, dec_preds))
MAE: 213498.91860465117
MSE: 158967825632.52325
RMSE: 398707.6944736874
R^2:  0.12351271172499989
#CV score
print('Cross validation score: ',-1 * ((cross_val_score(decTree, X, y, cv=10, scoring='neg_mean_squared_error')).mean()))
Cross validation score:  233772587653.33438
From the decision tree scores, it would be in 3rd in for in ranking our ML methods by predictive power.

Conclusions
From our various models metrics, we see that could not find a model with very high predictive power. Nonetheless, the best performing model with regard to the metrics we measured was our linear regression model. Since our RMSE for linear regression is actually less than our standard deviation for travelers in 2020, it does have a bit of predictive power. We do recognize that our model could use more features to aid in its predictive power. Some possible features to improve the model would be airfare prices, number of flights, travel bans, etc. As well, we recognize that the United States is a very big country and each part of it has been experiencing different trials throughout the year. From the pandemic to civil unrest, predicting events this year has been tough. Not only that, but we saw from our EDA that there was a sharp up tick of travelers near the end of year. This could be from people traveling home to be with their families, even though cases were higher than when a lot of the country had many shutdowns. Nonetheless, using country states of economic status, public status, recreation status, and CDC guideline of masks, we were able to construct a machine learning model to predict the number of air travelers on a given day in the United States.
