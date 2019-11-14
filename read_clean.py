import pandas as pd
import numpy as np

file = 'london_merged.tsv'
df = pd.read_csv(file, sep='\t')
print(df.info())

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
df.rename(columns={'timestamp':'date_time',
                        'cnt':'count',
                        't1':'temp',
                        't2':'tempf',
                        'hum':'humidity'},inplace=True)

# categorical variables
df['weather_code'] = df.season.astype('category')
df['season'] = df.season.astype('category')
df['is_holiday'] = df.season.astype('category')
df['is_weekend'] = df.season.astype('category')

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


# date time conversion
df['date_time'] = pd.to_datetime(df['date_time'], format ="%Y-%m-%d %H:%M:%S")

# separating month, day, and hour
df['month'] = df['date_time'].apply(lambda x : str(x).split(' ')[0].split('-')[1])
df['day'] = df['date_time'].apply(lambda x : str(x).split(' ')[0].split('-')[2])
df['hour'] = df['date_time'].apply(lambda x : str(x).split(' ')[1].split(':')[0])



# boxplots of bikeshare use by month, day, and hour
sns.boxplot(data=df, x='month', y='count')
plt.show()
sns.boxplot(data=df, x='day', y='count')
plt.show()
sns.boxplot(data=df, x='hour', y='count')
plt.show()

# barplots of total use on weekend, season, and holiday
sns.barplot(x=df['is_weekend'],y=df['count'])
plt.show()
sns.barplot(x=df['is_holiday'],y=df['count'])
plt.show()
sns.barplot(x=df['season'],y=df['count'])
plt.show()

