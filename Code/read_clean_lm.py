import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.regression.linear_model as sm
from statsmodels.tools.tools import add_constant

#::--------------------------------
# Read in the dataset
#::--------------------------------
file = 'london_merged.tsv'
df = pd.read_csv(file, sep='\t')
print(df.info())
print ('-'*80 + '\n')

#::--------------------------------
# Renaming the columns
#::--------------------------------
df.rename(columns={'cnt':'count',
                        't1':'temp',
                        't2':'tempf',
                        'hum':'humidity'},inplace=True)


#::--------------------------------
# Converting timestamp to datetime values
#::--------------------------------

df['timestamp'] = pd.to_datetime(df['timestamp'], format ="%Y-%m-%d %H:%M:%S")


# separating month, day, and hour
df['month'] = df['timestamp'].apply(lambda x : str(x).split(' ')[0].split('-')[1])
df['day'] = df['timestamp'].apply(lambda x : str(x).split(' ')[0].split('-')[2])
df['hour'] = df['timestamp'].apply(lambda x : str(x).split(' ')[1].split(':')[0])


#::--------------------------------
# Changing weather code to dummy variables
#::--------------------------------
# "weather_code" category description:
# 1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/fog in vicinity
# 2 = scattered clouds / few clouds
# 3 = Broken clouds
# 4 = Cloudy
# 7 = Rain/ light Rain shower/ Light rain
# 10 = rain with thunderstorm
# 26 = snowfall
# 94 = Freezing Fog


df['weather_code'] = df.weather_code.replace({1:'0', 2:'1', 3:'2', 4:'3',
                                         5:'4', 10:'5', 26:'6', 94:'7'})
# deal with this

#::--------------------------------
# Changing variable datatypes to 'category'
#::--------------------------------
df['weather_code'] = df.weather_code.astype('category')
df['season'] = df.season.astype('category')
df['is_holiday'] = df.is_holiday.astype('category')
df['is_weekend'] = df.is_weekend.astype('category')
df['month'] = df.month.astype('category')
df['day'] = df.day.astype('category')
df['hour'] = df.hour.astype('category')

df["count"] = np.log(df["count"]+1)


#::--------------------------------
# Exporting the cleaned dataset
#::--------------------------------]

export_csv = df.to_csv (r'C:\Users\carolinesklaver\Desktop\export_dataframe.csv',
                       index = None, header=True)


#::--------------------------------
# BASIC STATISTICS
#::--------------------------------
plt.hist(df["count"])
plt.title("Histogram of Bike Share Count")
plt.xlabel("Bike Count")
plt.ylabel("Frequency")
plt.show()


print('Are there any missing values?:')
print(df.isnull().values.any())
print ('-'*80 + '\n')
print(df.describe())
print ('-'*80 + '\n')

#::--------------------------------
# VISUAL PLOTS
#::--------------------------------

# boxplots of bikeshare use by month, day, and hour
sns.boxplot(data=df, x='month', y='count')
plt.show()
sns.boxplot(data=df, x='day', y='count')
plt.show()
sns.boxplot(data=df, x='hour', y='count')
plt.show()

# barplots of total use on weekend, season, and holiday
ax1 = sns.barplot(x=df['is_weekend'],y=df['count'])
ax1.set_xticklabels(["Weekday","Weekend"])
ax1.set_xlabel('')
ax1.set_ylabel('Bike Count')
plt.show()
ax2 = sns.barplot(x=df['is_holiday'],y=df['count'])
ax2.set_xticklabels(["Non-holiday","Holiday"])
ax2.set_xlabel('')
ax2.set_ylabel('Bike Count')
plt.show()
ax3 = sns.barplot(x=df['season'],y=df['count'])
ax3.set_xticklabels(["Spring","Summer", "Fall", "Winter"])
ax3.set_xlabel('')
ax3.set_ylabel('Bike Count')
plt.show()


#::--------------------------------
# CORRELATIONS
#::--------------------------------
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()




#::--------------------------------
# LINEAR REGRESSION
#::--------------------------------

# defining feature matrix(X) and response vector(y)
X = df.iloc[:,3:13]
y = df.iloc[:,1]

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=1)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# regression coefficients
print('Coefficients: \n', reg.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))


# Printing R2 and Mean Squared Error
print('R2 score: %.2f' % r2_score(y_test,y_pred)) # Priniting R2 Score
print('Mean squared Error :',mean_squared_error(y_test,y_pred))

print("Intercept:", reg.intercept_)
parameters = zip(X.columns, reg.coef_)
print("Parameters:", list(parameters))
RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE:", RMSE)



#::--------------------------------
# OLS function to see p-values, etc.
#::--------------------------------
result = sm.OLS(y.astype(float), add_constant(X).astype(float)).fit()
print(result.summary())  # month is not significant



#::--------------------------------
# PLOT RESIDUALS
#::--------------------------------
# plot residual error
## setting plot style
plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
plt.legend(loc='upper right')
plt.title("Residual errors")
plt.show()

