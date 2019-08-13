# Notebook 1: Exploratory Data Analysis and Data Cleaning
[![Notebook on exploratory data analysis and data cleaning.](https://github.com/michellekli/visitor-forecasting/blob/master/notebook1-preview.png)](https://github.com/michellekli/visitor-forecasting/blob/master/visitor-forecasting-part1.ipynb)

# Notebook 2: Data Preparation, Time Series Analysis, and Modeling
[![Notebook on data preparation, time series analysis, and modeling.](https://github.com/michellekli/visitor-forecasting/blob/master/notebook2-preview.png)](https://github.com/michellekli/visitor-forecasting/blob/master/visitor-forecasting-part2.ipynb)

# Introduction
Restaurants operate in the tricky business of serving perishable products. Accurately forecasting customer demand is essential for increasing profits, since it allows for reduction of food wastage and labor costs. With forecasts, managers can predict how much perishable inventory to keep on hand and predict how much food should be prepped at the beginning of each day. Managers can also schedule sufficient staff according to forecasts to ensure customer satisfaction during normal and peak hours. Increased scheduling efficiency also benefits the staff because managers can more easily ensure their staff are assigned enough shifts for their livelihood and are not being overworked.

This repo houses a two-notebook project on the [Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting) dataset provided by Recruit Holdings. In this project, I create Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors (SARIMAX) and Bayesian Structural Time Series (BSTS) models to forecast the daily number of visitors at restaurants. [Part 1](https://github.com/michellekli/visitor-forecasting/blob/master/visitor-forecasting-part1.ipynb) covers exploratory data analysis and data cleaning. [Part 2](https://github.com/michellekli/visitor-forecasting/blob/master/visitor-forecasting-part2.ipynb) covers data preparation, time series analysis, and modeling.

# Results
![30 day forecast.](https://github.com/michellekli/visitor-forecasting/blob/master/forecasts/figures/air_5c817ef28f236bdf_30days.png)

# Data Set

## Features
> ### air_reserve.csv
>
> This file contains reservations made in the air system. Note that the reserve_datetime indicates the time when the reservation was created, whereas the visit_datetime is the time in the future where the visit will occur.
>
> * air_store_id - the restaurant's id in the air system
> * visit_datetime - the time of the reservation
> * reserve_datetime - the time the reservation was made
> * reserve_visitors - the number of visitors for that reservation
>
> ### hpg_reserve.csv
>
> This file contains reservations made in the hpg system.
>
> * hpg_store_id - the restaurant's id in the hpg system
> * visit_datetime - the time of the reservation
> * reserve_datetime - the time the reservation was made
> * reserve_visitors - the number of visitors for that reservation
>
> ### air_store_info.csv
>
> This file contains information about select air restaurants. Column names and contents are self-explanatory.
>
> * air_store_id
> * air_genre_name
> * air_area_name
> * latitude
> * longitude
>
> Note: latitude and longitude are the latitude and longitude of the area to which the store belongs
>
> ### hpg_store_info.csv
>
> This file contains information about select hpg restaurants. Column names and contents are self-explanatory.
>
> * hpg_store_id
> * hpg_genre_name
> * hpg_area_name
> * latitude
> * longitude
>
> Note: latitude and longitude are the latitude and longitude of the area to which the store belongs
>
> ### store_id_relation.csv
>
> This file allows you to join select restaurants that have both the air and hpg system.
>
> * hpg_store_id
> * air_store_id
>
> ### air_visit_data.csv
>
> This file contains historical visit data for the air restaurants.
>
> * air_store_id
> * visit_date - the date
> * visitors - the number of visitors to the restaurant on the date
>
> ### sample_submission.csv
>
> This file shows a submission in the correct format, including the days for which you must forecast.
>
> * id - the id is formed by concatenating the air_store_id and visit_date with an underscore
> * visitors- the number of visitors forecasted for the store and date combination
>
> ### date_info.csv
>
> This file gives basic information about the calendar dates in the dataset.
>
> * calendar_date
> * day_of_week
> * holiday_flg - is the day a holiday in Japan
> 
> [Source: Recruit Holdings, Dataset Creator](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/data)
