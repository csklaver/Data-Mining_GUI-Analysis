###IMPORTING DATA###

import pandas as pd
import numpy as np

df = pd.read_csv('london_merged.csv')
print(df.info())
print ('-'*80 + '\n')

# "timestamp" - timestamp field for grouping the data
# "cnt" - the count of a new bike shares
# "t1" - real temperature in C
# "t2" - temperature in C "feels like"
# "hum" - humidity in percentage
# "wind_speed" - wind speed in km/h
# "weather_code" - category of the weather
# "is_holiday" - boolean field - 1 holiday / 0 non holiday
# "is_weekend" - boolean field - 1 if the day is weekend
# "season" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.


# Renaming columns names to more readable names
df.rename(columns={'timestamp':'Time','cnt':'Count','t1':'Temp','t2':'TempF','hum':'Humidity','wind_speed':'Speed','weather_code':'Weather Code','is_holiday':'Holiday?','is_weekend':'Weekend?','season':'Season'},inplace=True)

# date time conversion
df['Time'] = pd.to_datetime(df['Time'], format ="%Y-%m-%d %H:%M:%S")

# categorical variables
print('Data types before change:')
print(df.dtypes)
df['Weather Code'] = df['Weather Code'].astype('category')
df['Holiday?'] = df['Holiday?'].astype('category')
df['Weekend?'] = df['Weekend?'].astype('category')
df['Season'] = df['Season'].astype('category')
print('\n','Data types after change:')
print(df.dtypes)
print ('-'*80 + '\n')

# "weather_code" category description:
# 1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/fog in vicinity
# 2 = scattered clouds / few clouds
# 3 = Broken clouds
# 4 = Cloudy
# 7 = Rain/ light Rain shower/ Light rain
# 10 = rain with thunderstorm
# 26 = snowfall
# 94 = Freezing Fog

# "is_holiday" - boolean field - 1 holiday / 0 non holiday
# "is_weekend" - boolean field - 1 if the day is weekend
# "season" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.

###BASIC STATISTICS###
print('Are there any missing values?:')
print(df.isnull().values.any())
print ('-'*80 + '\n')
print(df.describe())
print ('-'*80 + '\n')
