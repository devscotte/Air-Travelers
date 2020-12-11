# Travel in 2020

The year 2020 has been filled with many different events that have impacted various commercial areas. Specifically, the travel industry in the United States has seen a significant down tick in the number of people traveling. The purpose of this project is to collect data from various different sources to see if we can predict with high accuracy the number of people that would travel on a single day in 2020, specifically flying. We will be using data for number of travelers from the TSA website, data from the ourworldindata.org for covid case counts per day, stock market data from yahoo finance, and various other sources to know dates of travel bans. We want to see if the economic status, public health status, sports status, or CDC guidelines can be used to predict with high accuracy the number of air travelers in the United States.

## Import Libraries


```python
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
```

## Read in data


```python
#Read in covid case and death count data
covid = pd.read_csv('daily-covid-cases-deaths.csv')
```


```python
#Check to see if read in properly
covid.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Entity</th>
      <th>Code</th>
      <th>Date</th>
      <th>Daily new confirmed cases of COVID-19</th>
      <th>Daily new confirmed deaths due to COVID-19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-01-23</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-01-24</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-01-25</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-01-26</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>2020-01-27</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Read in stock market data
stock = pd.read_csv('^DJI.csv')
```


```python
#Check to see if read in properly
stock.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-12-10</td>
      <td>27900.650391</td>
      <td>27949.019531</td>
      <td>27804.000000</td>
      <td>27881.720703</td>
      <td>27881.720703</td>
      <td>213250000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-12-11</td>
      <td>27867.310547</td>
      <td>27925.500000</td>
      <td>27801.800781</td>
      <td>27911.300781</td>
      <td>27911.300781</td>
      <td>213510000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-12-12</td>
      <td>27898.339844</td>
      <td>28224.949219</td>
      <td>27859.869141</td>
      <td>28132.050781</td>
      <td>28132.050781</td>
      <td>277740000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-12-13</td>
      <td>28123.640625</td>
      <td>28290.730469</td>
      <td>28028.320313</td>
      <td>28135.380859</td>
      <td>28135.380859</td>
      <td>250660000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-12-16</td>
      <td>28191.669922</td>
      <td>28337.490234</td>
      <td>28191.669922</td>
      <td>28235.890625</td>
      <td>28235.890625</td>
      <td>286770000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Read in data for number of travelers
tsa_data = pd.read_csv('tsa_data.csv')
```


```python
#Check to see if read in properly
tsa_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Total Traveler Throughput</th>
      <th>Total Traveler Throughput (1 Year Ago - Same Weekday)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12/9/2020</td>
      <td>564,372</td>
      <td>2,020,488</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12/8/2020</td>
      <td>501,513</td>
      <td>1,897,051</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12/7/2020</td>
      <td>703,546</td>
      <td>2,226,290</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12/6/2020</td>
      <td>837,137</td>
      <td>2,292,079</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12/5/2020</td>
      <td>629,430</td>
      <td>1,755,801</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Sources for NBA and other sport suspensions
# https://www.nba.com/news/nba-suspend-season-following-wednesdays-games
# https://bleacherreport.com/articles/2880569-timeline-of-coronavirus-impact-on-sports
```


```python
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
        
```


```python
nba.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Games</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Source for first mask recommendation made by CDC
# https://www.npr.org/sections/coronavirus-live-updates/2020/04/03/826219824/president-trump-says-cdc-now-recommends-americans-wear-cloth-masks-in-public
```


```python
#Create dataset to show when masks were recommended

masks = pd.DataFrame()

masks_lst = []

for i in range(1, 366):
    if i < 93:
        masks_lst.append(0)
    else:
        masks_lst.append(1)

masks['recommendation'] = masks_lst
```


```python
#Check dataset
masks
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recommendation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>360</th>
      <td>1</td>
    </tr>
    <tr>
      <th>361</th>
      <td>1</td>
    </tr>
    <tr>
      <th>362</th>
      <td>1</td>
    </tr>
    <tr>
      <th>363</th>
      <td>1</td>
    </tr>
    <tr>
      <th>364</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>365 rows × 1 columns</p>
</div>



## Clean up data and Create main dataframe


```python
#Clean up covid dataset to only include United States
usa_covid = covid.loc[covid['Entity'] == 'United States']
```


```python
#Check earliest date for tsa dates
tsa_data.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Total Traveler Throughput</th>
      <th>Total Traveler Throughput (1 Year Ago - Same Weekday)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>279</th>
      <td>3/5/2020</td>
      <td>2,130,015</td>
      <td>2,402,692</td>
    </tr>
    <tr>
      <th>280</th>
      <td>3/4/2020</td>
      <td>1,877,401</td>
      <td>2,143,619</td>
    </tr>
    <tr>
      <th>281</th>
      <td>3/3/2020</td>
      <td>1,736,393</td>
      <td>1,979,558</td>
    </tr>
    <tr>
      <th>282</th>
      <td>3/2/2020</td>
      <td>2,089,641</td>
      <td>2,257,920</td>
    </tr>
    <tr>
      <th>283</th>
      <td>3/1/2020</td>
      <td>2,280,522</td>
      <td>2,301,439</td>
    </tr>
  </tbody>
</table>
</div>



The earliest date for the flights begin on March 1, 2020. We will need to drop all other observations before that date.


```python
#Drop all dates from stock before March 1
stock = stock[(stock['Date'] >= '2020-03-01') & (stock['Date'] <= '2020-12-09')].reset_index()
```


```python
#Drop all dates from covid before March 1
usa_covid = usa_covid[(usa_covid['Date'] >= '2020-03-01') & (usa_covid['Date'] <= '2020-12-09')].reset_index()
```


```python
#Check to see that all worked out
stock.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>55</td>
      <td>2020-03-02</td>
      <td>25590.509766</td>
      <td>26706.169922</td>
      <td>25391.960938</td>
      <td>26703.320313</td>
      <td>26703.320313</td>
      <td>637200000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56</td>
      <td>2020-03-03</td>
      <td>26762.470703</td>
      <td>27084.589844</td>
      <td>25706.279297</td>
      <td>25917.410156</td>
      <td>25917.410156</td>
      <td>647080000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>57</td>
      <td>2020-03-04</td>
      <td>26383.679688</td>
      <td>27102.339844</td>
      <td>26286.310547</td>
      <td>27090.859375</td>
      <td>27090.859375</td>
      <td>457590000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58</td>
      <td>2020-03-05</td>
      <td>26671.919922</td>
      <td>26671.919922</td>
      <td>25943.330078</td>
      <td>26121.279297</td>
      <td>26121.279297</td>
      <td>477370000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>2020-03-06</td>
      <td>25457.210938</td>
      <td>25994.380859</td>
      <td>25226.619141</td>
      <td>25864.779297</td>
      <td>25864.779297</td>
      <td>599780000</td>
    </tr>
  </tbody>
</table>
</div>




```python
usa_covid.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Entity</th>
      <th>Code</th>
      <th>Date</th>
      <th>Daily new confirmed cases of COVID-19</th>
      <th>Daily new confirmed deaths due to COVID-19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>61551</td>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-01</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>61552</td>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-02</td>
      <td>23</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61553</td>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-03</td>
      <td>19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61554</td>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-04</td>
      <td>33</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>61555</td>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-05</td>
      <td>77</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Drop the index columns
stock.drop(labels = 'index', axis = 1, inplace = True)
usa_covid.drop(labels = 'index', axis = 1, inplace = True)
```


```python
#Check to see it worked
stock.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-03-02</td>
      <td>25590.509766</td>
      <td>26706.169922</td>
      <td>25391.960938</td>
      <td>26703.320313</td>
      <td>26703.320313</td>
      <td>637200000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-03-03</td>
      <td>26762.470703</td>
      <td>27084.589844</td>
      <td>25706.279297</td>
      <td>25917.410156</td>
      <td>25917.410156</td>
      <td>647080000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-03-04</td>
      <td>26383.679688</td>
      <td>27102.339844</td>
      <td>26286.310547</td>
      <td>27090.859375</td>
      <td>27090.859375</td>
      <td>457590000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-03-05</td>
      <td>26671.919922</td>
      <td>26671.919922</td>
      <td>25943.330078</td>
      <td>26121.279297</td>
      <td>26121.279297</td>
      <td>477370000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-03-06</td>
      <td>25457.210938</td>
      <td>25994.380859</td>
      <td>25226.619141</td>
      <td>25864.779297</td>
      <td>25864.779297</td>
      <td>599780000</td>
    </tr>
  </tbody>
</table>
</div>




```python
usa_covid.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Entity</th>
      <th>Code</th>
      <th>Date</th>
      <th>Daily new confirmed cases of COVID-19</th>
      <th>Daily new confirmed deaths due to COVID-19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-01</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-02</td>
      <td>23</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-03</td>
      <td>19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-04</td>
      <td>33</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-05</td>
      <td>77</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(usa_covid)
```




    284




```python
len(stock)
```




    198




```python
#Attempt to merge covid and stock
main_df = usa_covid.merge(stock, how = 'left', left_on = 'Date', right_on = 'Date')
```


```python
#Change the NBA dataframe and Mask dataframe to match the dates

nba = nba.iloc[60 : 344]
masks = masks.iloc[60 : 344]
```


```python
nba = nba.reset_index(drop = True)
```


```python
masks = masks.reset_index(drop = True)
```


```python
#Add masks and nba to the main dataframe
main_df['Sports'] = nba['Games']
main_df['Masks'] = masks['recommendation']
```


```python
#Check to see if it's all there
main_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Entity</th>
      <th>Code</th>
      <th>Date</th>
      <th>Daily new confirmed cases of COVID-19</th>
      <th>Daily new confirmed deaths due to COVID-19</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>Sports</th>
      <th>Masks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-01</td>
      <td>7</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-02</td>
      <td>23</td>
      <td>5</td>
      <td>25590.509766</td>
      <td>26706.169922</td>
      <td>25391.960938</td>
      <td>26703.320313</td>
      <td>26703.320313</td>
      <td>637200000.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-03</td>
      <td>19</td>
      <td>1</td>
      <td>26762.470703</td>
      <td>27084.589844</td>
      <td>25706.279297</td>
      <td>25917.410156</td>
      <td>25917.410156</td>
      <td>647080000.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-04</td>
      <td>33</td>
      <td>4</td>
      <td>26383.679688</td>
      <td>27102.339844</td>
      <td>26286.310547</td>
      <td>27090.859375</td>
      <td>27090.859375</td>
      <td>457590000.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>United States</td>
      <td>USA</td>
      <td>2020-03-05</td>
      <td>77</td>
      <td>1</td>
      <td>26671.919922</td>
      <td>26671.919922</td>
      <td>25943.330078</td>
      <td>26121.279297</td>
      <td>26121.279297</td>
      <td>477370000.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Drop entity and code
main_df.drop(labels = ['Entity', 'Code'], axis = 1, inplace = True)
```


```python
main_df.rename({'Daily new confirmed cases of COVID-19' : 'Cases', 'Daily new confirmed deaths due to COVID-19' : 'Deaths'}, axis = 1, inplace = True)
```


```python
#Replace NANs 
main_df.fillna({'Open' : main_df['Open'].median(), 
               'High' : main_df['High'].median(),
               'Low' : main_df['Low'].median(), 
               'Close' : main_df['Close'].median(),
               'Adj Close' : main_df['Adj Close'].median(),
               'Volume' : main_df['Volume'].median()}, inplace = True)
```


```python
#Check it out
main_df.isna().sum()
```




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




```python
#Finally add the tsa dataframe
main_df['Travelers 2020'] = tsa_data['Total Traveler Throughput']
main_df['Travelers 2019'] = tsa_data['Total Traveler Throughput (1 Year Ago - Same Weekday)']
```


```python
#Look at final product?
main_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Cases</th>
      <th>Deaths</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>Sports</th>
      <th>Masks</th>
      <th>Travelers 2020</th>
      <th>Travelers 2019</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-03-01</td>
      <td>7</td>
      <td>0</td>
      <td>26668.264649</td>
      <td>26855.940429</td>
      <td>26500.794922</td>
      <td>26668.174805</td>
      <td>26668.174805</td>
      <td>401285000.0</td>
      <td>1</td>
      <td>0</td>
      <td>564,372</td>
      <td>2,020,488</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-03-02</td>
      <td>23</td>
      <td>5</td>
      <td>25590.509766</td>
      <td>26706.169922</td>
      <td>25391.960938</td>
      <td>26703.320313</td>
      <td>26703.320313</td>
      <td>637200000.0</td>
      <td>1</td>
      <td>0</td>
      <td>501,513</td>
      <td>1,897,051</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-03-03</td>
      <td>19</td>
      <td>1</td>
      <td>26762.470703</td>
      <td>27084.589844</td>
      <td>25706.279297</td>
      <td>25917.410156</td>
      <td>25917.410156</td>
      <td>647080000.0</td>
      <td>1</td>
      <td>0</td>
      <td>703,546</td>
      <td>2,226,290</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-03-04</td>
      <td>33</td>
      <td>4</td>
      <td>26383.679688</td>
      <td>27102.339844</td>
      <td>26286.310547</td>
      <td>27090.859375</td>
      <td>27090.859375</td>
      <td>457590000.0</td>
      <td>1</td>
      <td>0</td>
      <td>837,137</td>
      <td>2,292,079</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-03-05</td>
      <td>77</td>
      <td>1</td>
      <td>26671.919922</td>
      <td>26671.919922</td>
      <td>25943.330078</td>
      <td>26121.279297</td>
      <td>26121.279297</td>
      <td>477370000.0</td>
      <td>1</td>
      <td>0</td>
      <td>629,430</td>
      <td>1,755,801</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>279</th>
      <td>2020-12-05</td>
      <td>213881</td>
      <td>2254</td>
      <td>26668.264649</td>
      <td>26855.940429</td>
      <td>26500.794922</td>
      <td>26668.174805</td>
      <td>26668.174805</td>
      <td>401285000.0</td>
      <td>1</td>
      <td>1</td>
      <td>2,130,015</td>
      <td>2,402,692</td>
    </tr>
    <tr>
      <th>280</th>
      <td>2020-12-06</td>
      <td>175664</td>
      <td>1113</td>
      <td>26668.264649</td>
      <td>26855.940429</td>
      <td>26500.794922</td>
      <td>26668.174805</td>
      <td>26668.174805</td>
      <td>401285000.0</td>
      <td>1</td>
      <td>1</td>
      <td>1,877,401</td>
      <td>2,143,619</td>
    </tr>
    <tr>
      <th>281</th>
      <td>2020-12-07</td>
      <td>192435</td>
      <td>1404</td>
      <td>30233.029297</td>
      <td>30233.029297</td>
      <td>29967.220703</td>
      <td>30069.789063</td>
      <td>30069.789063</td>
      <td>365810000.0</td>
      <td>1</td>
      <td>1</td>
      <td>1,736,393</td>
      <td>1,979,558</td>
    </tr>
    <tr>
      <th>282</th>
      <td>2020-12-08</td>
      <td>215878</td>
      <td>2546</td>
      <td>29997.949219</td>
      <td>30246.220703</td>
      <td>29972.070313</td>
      <td>30173.880859</td>
      <td>30173.880859</td>
      <td>311190000.0</td>
      <td>1</td>
      <td>1</td>
      <td>2,089,641</td>
      <td>2,257,920</td>
    </tr>
    <tr>
      <th>283</th>
      <td>2020-12-09</td>
      <td>221267</td>
      <td>3124</td>
      <td>30229.810547</td>
      <td>30319.699219</td>
      <td>29951.849609</td>
      <td>30068.810547</td>
      <td>30068.810547</td>
      <td>380520000.0</td>
      <td>1</td>
      <td>1</td>
      <td>2,280,522</td>
      <td>2,301,439</td>
    </tr>
  </tbody>
</table>
<p>284 rows × 13 columns</p>
</div>



Our dataframe is all prepared and ready to do some EDA.

## EDA


```python
#Look at summary statistics
main_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cases</th>
      <th>Deaths</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>Sports</th>
      <th>Masks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>284.000000</td>
      <td>284.000000</td>
      <td>284.000000</td>
      <td>284.000000</td>
      <td>284.000000</td>
      <td>284.000000</td>
      <td>284.000000</td>
      <td>2.840000e+02</td>
      <td>284.000000</td>
      <td>284.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>54177.947183</td>
      <td>1018.915493</td>
      <td>26390.153733</td>
      <td>26619.178649</td>
      <td>26157.870468</td>
      <td>26392.210869</td>
      <td>26392.210869</td>
      <td>4.298291e+08</td>
      <td>0.584507</td>
      <td>0.887324</td>
    </tr>
    <tr>
      <th>std</th>
      <td>48470.687826</td>
      <td>649.333891</td>
      <td>2063.128829</td>
      <td>1986.881384</td>
      <td>2152.148505</td>
      <td>2064.710643</td>
      <td>2064.710643</td>
      <td>1.117051e+08</td>
      <td>0.493677</td>
      <td>0.316755</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>19028.359375</td>
      <td>19121.009766</td>
      <td>18213.650391</td>
      <td>18591.929688</td>
      <td>18591.929688</td>
      <td>1.770400e+08</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25213.500000</td>
      <td>537.250000</td>
      <td>25900.005859</td>
      <td>26102.789551</td>
      <td>25609.060058</td>
      <td>25855.424317</td>
      <td>25855.424317</td>
      <td>3.733825e+08</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>40737.000000</td>
      <td>940.500000</td>
      <td>26668.264649</td>
      <td>26855.940429</td>
      <td>26500.794922</td>
      <td>26668.174805</td>
      <td>26668.174805</td>
      <td>4.012850e+08</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>61185.250000</td>
      <td>1295.500000</td>
      <td>27542.353027</td>
      <td>27810.099610</td>
      <td>27447.347656</td>
      <td>27602.400391</td>
      <td>27602.400391</td>
      <td>4.402475e+08</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>227828.000000</td>
      <td>3124.000000</td>
      <td>30233.029297</td>
      <td>30319.699219</td>
      <td>29989.560547</td>
      <td>30218.259766</td>
      <td>30218.259766</td>
      <td>9.082600e+08</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Change Travelers data type
for i in range(0, len(main_df)):
    main_df['Travelers 2020'][i] = main_df['Travelers 2020'][i].replace(',', '')
    main_df['Travelers 2019'][i] = main_df['Travelers 2019'][i].replace(',', '')
```

    C:\Users\thebs\anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    C:\Users\thebs\anaconda3\lib\site-packages\ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    


```python
main_df['Travelers 2020'] = pd.to_numeric(main_df['Travelers 2020'])
main_df['Travelers 2019'] = pd.to_numeric(main_df['Travelers 2019'])
```


```python
#All summary statistics
main_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cases</th>
      <th>Deaths</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>Sports</th>
      <th>Masks</th>
      <th>Travelers 2020</th>
      <th>Travelers 2019</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>284.000000</td>
      <td>284.000000</td>
      <td>284.000000</td>
      <td>284.000000</td>
      <td>284.000000</td>
      <td>284.000000</td>
      <td>284.000000</td>
      <td>2.840000e+02</td>
      <td>284.000000</td>
      <td>284.000000</td>
      <td>2.840000e+02</td>
      <td>2.840000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>54177.947183</td>
      <td>1018.915493</td>
      <td>26390.153733</td>
      <td>26619.178649</td>
      <td>26157.870468</td>
      <td>26392.210869</td>
      <td>26392.210869</td>
      <td>4.298291e+08</td>
      <td>0.584507</td>
      <td>0.887324</td>
      <td>6.383851e+05</td>
      <td>2.375686e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>48470.687826</td>
      <td>649.333891</td>
      <td>2063.128829</td>
      <td>1986.881384</td>
      <td>2152.148505</td>
      <td>2064.710643</td>
      <td>2064.710643</td>
      <td>1.117051e+08</td>
      <td>0.493677</td>
      <td>0.316755</td>
      <td>4.012484e+05</td>
      <td>2.461785e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>19028.359375</td>
      <td>19121.009766</td>
      <td>18213.650391</td>
      <td>18591.929688</td>
      <td>18591.929688</td>
      <td>1.770400e+08</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.753400e+04</td>
      <td>1.591158e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25213.500000</td>
      <td>537.250000</td>
      <td>25900.005859</td>
      <td>26102.789551</td>
      <td>25609.060058</td>
      <td>25855.424317</td>
      <td>25855.424317</td>
      <td>3.733825e+08</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.466970e+05</td>
      <td>2.193565e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>40737.000000</td>
      <td>940.500000</td>
      <td>26668.264649</td>
      <td>26855.940429</td>
      <td>26500.794922</td>
      <td>26668.174805</td>
      <td>26668.174805</td>
      <td>4.012850e+08</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>6.339280e+05</td>
      <td>2.430741e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>61185.250000</td>
      <td>1295.500000</td>
      <td>27542.353027</td>
      <td>27810.099610</td>
      <td>27447.347656</td>
      <td>27602.400391</td>
      <td>27602.400391</td>
      <td>4.402475e+08</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.982395e+05</td>
      <td>2.556494e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>227828.000000</td>
      <td>3124.000000</td>
      <td>30233.029297</td>
      <td>30319.699219</td>
      <td>29989.560547</td>
      <td>30218.259766</td>
      <td>30218.259766</td>
      <td>9.082600e+08</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.280522e+06</td>
      <td>2.882915e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
```


```python
#Graph travel numbers from this year and last year
sns.jointplot(x = 'Travelers 2020', y = 'Travelers 2019', data = main_df)
```




    <seaborn.axisgrid.JointGrid at 0x177d2a53b88>




![png](output_49_1.png)


As we can clearly see, the numbers for 2020 have been significantly smaller than from 2019


```python
#Graph travelers 2020 vs cases
sns.lmplot(x = 'Travelers 2020', y = 'Cases', data = main_df)
```




    <seaborn.axisgrid.FacetGrid at 0x177d40e7c08>




![png](output_51_1.png)


Very interesting shape to this...  


```python
#Plot line graph of number of cases
sns.relplot(x = 'Date', y = 'Cases', kind = 'line', data = main_df)
```




    <seaborn.axisgrid.FacetGrid at 0x177d0f100c8>




![png](output_53_1.png)



```python
#Plot count of days with masks vs no masks
sns.countplot(x = 'Masks', data = main_df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x177d2123988>




![png](output_54_1.png)



```python
#Plot number of travelers vs number of cases
plt.figure(figsize=(10,6))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.plot('Date', 'Travelers 2020', data = main_df, label = 'Travelers')
plt.plot('Date', 'Cases', data = main_df, label = 'Cases')

plt.legend()
```




    <matplotlib.legend.Legend at 0x177d901ac88>




![png](output_55_1.png)


## Models

We will now fit our data to different machine learning models. We will be fitting the data to a linear, KNN, ridge, and decision tree model. We will run MAE, MSE, RMSE, and CV sc

## Linear Regression


```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
```


```python
#Separate response from explanatory
y = main_df['Travelers 2020']

X = main_df[['Cases', 'Deaths', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sports', 'Masks', 'Travelers 2019']]
```


```python
#Separate training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```


```python
lm = LinearRegression()
```


```python
#Fit a linear regression model
lm.fit(X_train,y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
#Show coefficients
print('Coefficients: \n', lm.coef_)
```

    Coefficients: 
     [ 4.34770099e+00  1.83464093e+02  1.17004012e+02 -1.82928386e+02
      6.75991449e+01 -3.32849264e+00 -3.32849263e+00  5.09455846e-05
     -3.70770896e+05 -5.31593389e+05  2.44579269e-01]
    


```python
#Gather predictions from fitted model
predictions = lm.predict(X_test)
```


```python
#Plot predictions vs actual
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
```




    Text(0, 0.5, 'Predicted Y')




![png](output_66_1.png)



```python
#Generate metrics
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('R^2: ', metrics.r2_score(y_test, predictions))
```

    MAE: 209582.34029446437
    MSE: 89312463686.86139
    RMSE: 298851.9092909754
    R^2:  0.5075655165151798
    


```python
#CV score
print('Cross validation score: ',-1 * ((cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')).mean()))
```

    Cross validation score:  279372589873.1498
    

With a RMSE of 298851 and CV score of 279372589873.1498, I would say that this model isn't the best for predicting the number of airplane travelers in 2020, but 2020 has been hard for anyone to predict so it's not the worst thing to come from 2020.


```python
#Plot residuals
sns.distplot((y_test-predictions),bins=60);
```


![png](output_70_0.png)


At least the residuals look normal? That's good? Our data really isn't that linear though, so linear regression assumptions are iffy on this data


```python
np.std(main_df['Travelers 2020'])
```




    400541.3894473091



## KNN


```python
from sklearn.neighbors import KNeighborsRegressor
```


```python
#KNN with 5
knn = KNeighborsRegressor(n_neighbors=5)
```


```python
#Fit the model
knn.fit(X_train, y_train)
```




    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                        metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                        weights='uniform')




```python
#Make predictions
knn_preds = knn.predict(X_test)
```


```python
#Get metrics
print('MAE:', metrics.mean_absolute_error(y_test, knn_preds))
print('MSE:', metrics.mean_squared_error(y_test, knn_preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, knn_preds)))
```

    MAE: 317331.5046511628
    MSE: 198213011157.18558
    RMSE: 445211.19837351976
    


```python
#Get CV score
print('Cross validation score: ',-1 * ((cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error')).mean()))
```

    Cross validation score:  230145124472.75333
    

Well, this model did not do any better at predicting with accuracy the amount of travelers flying on a given day in 2020 since the RMSE is much higher than that of the lm model. Let's try to find a proper amount of K with GridSearch.


```python
from sklearn.model_selection import GridSearchCV
```


```python
parameters = {'n_neighbors':list(range(1,30))}
knn = KNeighborsRegressor()
clf = GridSearchCV(knn, parameters)
clf.fit(X, y)
```




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




```python
clf.best_params_
```




    {'n_neighbors': 28}




```python
knn = KNeighborsRegressor(n_neighbors=28)
```


```python
knn.fit(X_train, y_train)
```




    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                        metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                        weights='uniform')




```python
knn_preds = knn.predict(X_test)
```


```python
print('MAE:', metrics.mean_absolute_error(y_test, knn_preds))
print('MSE:', metrics.mean_squared_error(y_test, knn_preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, knn_preds)))
```

    MAE: 317331.5046511628
    MSE: 198213011157.18558
    RMSE: 445211.19837351976
    

Looks like we get the same results. KNN is not very good at and not as good as linear regression.

## Ridge


```python
from sklearn.linear_model import Ridge
```


```python
ridge = Ridge()
```


```python
ridge.fit(X_train, y_train)
```

    C:\Users\thebs\anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=2.12647e-18): result may not be accurate.
      overwrite_a=True).T
    




    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)




```python
ridge_preds = ridge.predict(X_test)
```


```python
print('MAE:', metrics.mean_absolute_error(y_test, ridge_preds))
print('MSE:', metrics.mean_squared_error(y_test, ridge_preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, ridge_preds)))
print('R^2: ', metrics.r2_score(y_test, ridge_preds))
```

    MAE: 211176.6644662734
    MSE: 90417914812.4394
    RMSE: 300695.7179815493
    R^2:  0.5014704853003841
    


```python
#CV score
print('Cross validation score: ',-1 * ((cross_val_score(ridge, X, y, cv=10, scoring='neg_mean_squared_error')).mean()))
```

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

## Decision Tree Regressor


```python
from sklearn.tree import DecisionTreeRegressor
```


```python
decTree = DecisionTreeRegressor()
```


```python
decTree.fit(X_train, y_train)
```




    DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=None,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, presort='deprecated',
                          random_state=None, splitter='best')




```python
dec_preds = decTree.predict(X_test)
```


```python
print('MAE:', metrics.mean_absolute_error(y_test, dec_preds))
print('MSE:', metrics.mean_squared_error(y_test, dec_preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dec_preds)))
print('R^2: ', metrics.r2_score(y_test, dec_preds))
```

    MAE: 213498.91860465117
    MSE: 158967825632.52325
    RMSE: 398707.6944736874
    R^2:  0.12351271172499989
    


```python
#CV score
print('Cross validation score: ',-1 * ((cross_val_score(decTree, X, y, cv=10, scoring='neg_mean_squared_error')).mean()))
```

    Cross validation score:  233772587653.33438
    

From the decision tree scores, it would be in 3rd in for in ranking our ML methods by predictive power.

# Conclusions

From our various models metrics, we see that could not find a model with very high predictive power. Nonetheless, the best performing model with regard to the metrics we measured was our linear regression model. Since our RMSE for linear regression is actually less than our standard deviation for travelers in 2020, it does have a bit of predictive power. We do recognize that our model could use more features to aid in its predictive power. Some possible features to improve the model would be airfare prices, number of flights, travel bans, etc. As well, we recognize that the United States is a very big country and each part of it has been experiencing different trials throughout the year. From the pandemic to civil unrest, predicting events this year has been tough. Not only that, but we saw from our EDA that there was a sharp up tick of travelers near the end of year. This could be from people traveling home to be with their families, even though cases were higher than when a lot of the country had many shutdowns. Nonetheless, using country states of economic status, public status, recreation status, and CDC guideline of masks, we were able to construct a machine learning model to predict the number of air travelers on a given day in the United States. 


```python

```
