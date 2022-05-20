'''
#@Time      :4/26/22 19:36
#@Author    : Chelsea Li
#@File      :final project.py
#@Software  :PyCharm
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from lifelines import KaplanMeierFitter
from numpy import linalg as LA
from scipy.stats import iqr
import statistics as stats
from statsmodels.tsa.seasonal import STL
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from scipy import signal
import scipy
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker
from Toolbox import *
import warnings
warnings.filterwarnings("ignore")



# read in data
df_sale = pd.read_csv('Data/train.csv',parse_dates=['date'])
holidays = pd.read_csv('Data/holidays_events.csv',parse_dates=['date'])
oil_price = pd.read_csv('Data/oil.csv',parse_dates=['date'])
transations = pd.read_csv('Data/transactions.csv',parse_dates=['date'])

# df3=pd.DataFrame(index=df['date'],column='sales',data=df.sales.value)
# df_sale = df_sale.set_index('date').to_period('D')

# missing value check
holidays.info()
oil_price.info()
transations.info()

oilprice_missing_num = sum(oil_price.dcoilwtico.isna())
oilprice_missing_date = oil_price[oil_price.dcoilwtico.isna()]


# subset dataset  for automotive sales
df_auto = df_sale[df_sale.family == 'AUTOMOTIVE'].groupby('date').mean()['sales']
df_transations = transations.groupby('date').mean()['transactions']



## impute missing data of oil price with the oil price last day

imputer = KNNImputer(n_neighbors=3, weights="uniform")
oil_price.dcoilwtico = imputer.fit_transform(oil_price.dcoilwtico.values.reshape(-1,1))


# plot dependent variable - sales
plt.figure(figsize=(11, 8))
df_auto.plot()
plt.title('The time plot of Automotive Sales')
plt.xlabel('Date')
plt.ylabel('Automotive Sales Vs time ')
plt.show()

# check outliners
plt.figure()
plt.boxplot(df_auto)
plt.show()

# clean data
iqr = iqr(df_auto)
Q1 = np.percentile(df_auto,25)
Q3 = np.percentile(df_auto,75)
auto_mean = df_auto.mean()

def process_outliners(x,Q1,Q3,iqr,auto_mean):
    if x < Q1-1.5*iqr:
        return auto_mean
    elif x > Q3 + 1.5*iqr:
        return auto_mean
    else:
        return x

df_auto = df_auto.apply(lambda x : process_outliners(x,Q1,Q3,iqr,auto_mean))


# plot dependent variable - sales
plt.figure()
plt.boxplot(df_auto)
plt.show()


plt.figure(figsize=(11, 8))
df_auto.plot()
plt.title('The time plot of Automotive Sales after cleaning')
plt.xlabel('Date')
plt.ylabel('Automotive Sales Vs time ')
plt.show()


# prepare the dataset
df = pd.merge(df_auto,df_transations,how='left', on='date')
df = pd.merge(df,holidays[['date','type','locale']],how='left', on='date')
df = pd.merge(df,oil_price,how='left', on='date')

# impute missing values
df.info()

# transactions
df.transactions[df.transactions.isna()]
df.transactions = df.transactions.fillna(df.transactions.mean())


# oil price
imputer = KNNImputer(n_neighbors=3, weights="uniform")
df.dcoilwtico = imputer.fit_transform(df.dcoilwtico.values.reshape(-1,1))

# holiday
df.type = df.type.fillna('Non-holiday')
df.locale = df.locale.fillna('Non-holiday')


# categorical features encoder
df_data = pd.get_dummies(df,columns=['type','locale'])
df_data = df_data.drop(columns = 'date')

##################################### multi-linear regression model #################################
# Correlation matrix and
cor = df_data.corr()
# sns.set(rc = {'figure.figsize':(15,15)})
ax = sns.heatmap(cor,annot=True)
plt.show()



# SVD
H = np.matmul(df_data.T,df_data)
_,d,_ = np.linalg.svd(H)
print(f'singular values of X are {d}')
print(f'The condition number for X is {LA.cond(df_data)}')


#  OLS function
X_train, X_test,y_train,y_test = train_test_split(df_data.iloc[:,1:],df_data['sales'],test_size=0.2)
model = sm.OLS(y_train,X_train).fit()
print(model.summary())

#  backwards forecast regression
X_train_copy= X_train
X_train_copy = X_train_copy.drop(labels = 'type_Work Day',axis = 1)
model1 = sm.OLS(y_train,X_train_copy).fit()
print(model1.summary())


X_train_copy = X_train_copy.drop(labels = 'type_Event',axis = 1)
model2 = sm.OLS(y_train,X_train_copy).fit()
print(model2.summary())


X_train_copy = X_train_copy.drop(labels = 'type_Bridge',axis = 1)
model3 = sm.OLS(y_train,X_train_copy).fit()
print(model3.summary())


X_train_copy = X_train_copy.drop(labels = 'locale_Regional',axis = 1)
model4 = sm.OLS(y_train,X_train_copy).fit()
print(model4.summary())

X_train_copy = X_train_copy.drop(labels = 'locale_Local',axis = 1)
model5 = sm.OLS(y_train,X_train_copy).fit()
print(model5.summary())


# predict
X_test_copy = X_test
X_test_copy = X_test_copy.drop(labels =(['type_Work Day','type_Event','type_Bridge'
              ,'locale_Regional','locale_Local']) ,axis = 1)

y_test_forecast = model5.predict(X_test_copy)
y_train_predict = model5.predict(X_train_copy)


# residual analysis
residual1  = y_train.values - y_train_predict
fig = plt.figure()
plot_acf(residual1, ax=plt.gca(), lags=20)
plt.show()



forecast_error = y_test.values - y_test_forecast
fig = plt.figure()
plot_acf(forecast_error, ax=plt.gca(), lags=20)
plt.show()


# Q value
acf = sm.tsa.acf(residual1)
Q = len(y_train) * np.sum(np.square(acf[20:]))

# F-test(intercept only equal to model), t_test(Ho: beta equals to zero)
scipy.stats.ttest_ind(y_train,y_train_predict)
scipy.stats.f_oneway(y_train,y_train_predict)


# variance and mean of residual
print(f'The variance of residual is {np.var(residual1)}.')
print(f'The mean of residual is {np.mean(residual1)}.')




########################################### Base model forecast #################################################################
# 9 holt-winters method
############################ bug
df_auto_train, df_auto_test = train_test_split(df_auto, shuffle= False, test_size=0.2)

import statsmodels.tsa.holtwinters as ets
holt_t = ets.ExponentialSmoothing(df_auto_train,seasonal_periods=7).fit()
holt_f = holt_t.forecast(steps = len(df_auto_test))


# from statsmodels.tsa.api import Holt
# fitted = Holt(df_auto_train.fit(smoothing_level=0.5,smoothing_trend=0.2))
# yh_Holt_forecast = fitted.forecast(len(df_auto))
# yh_holt_train_forecast = fitted.fittedvalues


# 10 Base model - (average, naïve, drift, simple and exponential smoothing)
# Average method
yh_ave = np.mean(df_auto_train)
yh_ave_test_forecast = [yh_ave] * len(df_auto_test)
yh_ave_train_predict = Cal_rolling_Mean(df_auto_train)

# naive
yh_naive = df_auto_train.iloc[-1]
yh_naive_test_forecast = [yh_naive] * len(df_auto_test)

yh_naive_train_predict = [df_auto_train[0]]
for i in range(len(df_auto_train)-1):
    yh_naive_train_predict.append(df_auto_train[i])


# drift
def drift_method_equation(df_train,df_test):
    yh_drift_test_forecast = []
    yh_drift_train_predict = [df_train[0]]
    for i in range(1,len(df_train)):
        y_first = df_train[0]
        y_last = df_train[i]
        k = (y_last - y_first) / i
        b = y_first - (y_last - y_first) / i

        yh = k * (i + 1) + b
        yh_drift_train_predict.append(yh)

    for j in range(len(df_test)):
        yh = k * (len(df_train)+j+1) + b
        yh_drift_test_forecast.append(yh)

    return  yh_drift_test_forecast,yh_drift_train_predict

yh_drift_test_forecast,yh_drift_train_predict = drift_method_equation(df_auto_train,df_auto_test)


# SES
def SES_method(y_train,y_test,alpha):
    y_forecast = [y_train[0]]
    yt = y_train[0]
    for t in range(1,len(y_train)):
        yt = alpha * y_train[t]+(1-alpha)*yt
        y_forecast.append(yt)
    for i in range(len(y_test)):
        y_forecast.append(yt)
    y_test_forecast = [yt] * len(y_test)
    yh_train_forecast = y_forecast[:len(y_train)]
    return y_forecast, y_test_forecast, yh_train_forecast


yh_SES_whole_forecast,yh_SES_test_forecast,yh_SES_train_predict = SES_method(df_auto_train,df_auto_test, 0.5)


# plot base models
### train and test forecast dataset
df_train_predict = {'Average Method': yh_ave_train_predict,
              'Naive Method':yh_naive_train_predict,
              'Drift Method':yh_drift_train_predict,
              'SES Method(Alpha=0.5)':yh_SES_train_predict}

train_prediction_dataset = pd.DataFrame(df_train_predict)


df_test_forecast = {'Average Method': yh_ave_test_forecast,
              'Naive Method':yh_naive_test_forecast,
              'Drift Method':yh_drift_test_forecast,
              'SES Method(Alpha=0.5)':yh_SES_test_forecast}

test_forecast_dataset = pd.DataFrame(df_test_forecast)


plt.figure(figsize=(12,8))
for i, col in enumerate(test_forecast_dataset):
    plt.subplot(2,2,i+1)
    plt.plot(df_auto_train, label='train set')
    plt.plot(df_auto_test, label='Test data')
    #plt.plot(df_auto_train.index,df_train_predict[col],label=col)
    plt.plot(df_auto_test.index,test_forecast_dataset[col], label=col)
    plt.grid()
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'The plot of {col}')
plt.tight_layout()
plt.show()


# 3 error and variance
# train predict dataset
def prediction_error(df_train,df_train_forecast):
    error = abs(df_train - df_train_forecast)
    return error

def forecast_error(df_test,df_test_forecast):
    error = abs(df_test-df_test_forecast)
    return error

def Method_MSE(df,df_forecast):
    error_square = 0
    for i in range(len(df)):
        error_square += (df[i]-df_forecast[i])**2
    mse = np.mean(error_square)
    return mse


for idx in range(len(train_prediction_dataset.columns)):
    # predict and forecast error
    prediction_error_train = prediction_error(df_auto_train.values,train_prediction_dataset[train_prediction_dataset.columns[idx]])
    forecast_error_test = forecast_error(df_auto_test.values,test_forecast_dataset[test_forecast_dataset.columns[idx]])
    # MSE
    prediction_MSE = Method_MSE(df_auto_train.values,prediction_error_train)
    forecast_MSE = Method_MSE(df_auto_test.values,forecast_error_test)
    # Variance of error
    error_prediction_var = stats.variance(prediction_error_train)
    error_forecast_var = stats.variance(forecast_error_test)

    print(f'The {train_prediction_dataset.columns[idx]} MES of prediction error '
          f' is {np.round(prediction_MSE,2)}'
          f' and the MSE of forecast error is {np.round(forecast_MSE,2)}.')

    print(f'The {train_prediction_dataset.columns[idx]} method variance of prediction error '
          f' is {np.round(error_prediction_var,2)}'
          f' and the variance of forecast error is {np.round(error_forecast_var,2)}.')


# time series decomposition
# df_auto_2 = pd.Series(np.array(df_auto).reshape(len(df_auto)),index = df_auto.index)
STL = STL(df_auto,period=7)
res = STL.fit()
fig = res.plot()
plt.show()


T = res.trend
R = res.resid

F = np.maximum(0,1-np.var(np.array(R))/np.var(np.array(T)+np.array(R)))
print(f'Strength of Trend is {F}')


S = res.seasonal
F = np.maximum(0,1-np.var(np.array(R))/np.var(np.array(S)+np.array(R)))
print(f'Strength of seasonality is {F}')



# 4  plot detrended and seasonality adjusted dataset
R = res.trend
trend_adj = df_auto.values - R.values
plt.figure(figsize=(11, 8))
plt.plot(df_auto)
plt.plot(pd.DataFrame(trend_adj,index = df_auto.index))
plt.title('Detrended Dataset')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(25))
plt.xticks(rotation=60)
plt.grid()
plt.xlabel('Date')
plt.ylabel('Automotive Sales Vs time ')
plt.show()



S = res.seasonal
trend_seasonal_adj = df_auto.values - S.values
plt.figure(figsize=(11, 8))
plt.plot(df_auto)
plt.plot(pd.DataFrame(trend_seasonal_adj ,index = df_auto.index))
plt.title('Seasonality Adjusted dataset')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(25))
plt.xticks(rotation=60)
plt.grid()
plt.xlabel('Date')
plt.ylabel('Automotive Sales Vs time ')
plt.show()


####################################### ARMA, ARIMA, SARIMA ###################################
# ACF PACF - dependent variable
ACF_PACF_Plot(df_auto,50)


# 2 Stationary
Cal_rolling_Mean_Var(df_auto,condision='without difference')
ADF_Cal(df_auto)
kpss_test(df_auto)


## difference
df_auto_diff = difference(df_auto,interval=7)
Cal_rolling_Mean_Var(df_auto_diff,condision='without difference')
ADF_Cal(df_auto_diff)
kpss_test(df_auto_diff)

ACF_PACF_Plot(df_auto_diff,50)


# GAPC
acf = sm.tsa.stattools.acf(df_auto_diff)
df = pd.DataFrame(GPAC(10,acf),columns=range(1,11))
ax = sns.heatmap(df,annot=True)
plt.show()



# LM package, sd and confidence interval

new_seta,cov,SSE_record = LM_algorithm(df_auto_train,7,8)
seta_std1 = np.std(new_seta)

def confidence_interval(seta,na,nb,cov):
    n = na + nb
    for i in range(n):
        low = seta[i] + 2*(cov[i][i]**(0.5))
        high = seta[i] - 2*(cov[i][i]**(0.5))
        print(f'The confidence interval of {i+1} parameter is {[low,high]}.')

confidence_interval(new_seta,7,8,cov)


seta1_1 = np.array([0,0,0,0,0,0,-1, 0.402,  0,  0,  0,  0, 0, -0.708,-0.129])

# 15 Diagnostic analysis (chi-square test, zero_pole_cancellation)
# one step prediction

# y(t) - y(t-7) = e(t) + 0.33e(t-1) -  0.747e(t-7) - 0.169e(t-8)
y_hat_t_1 = []
for i in range(6, len(df_auto_train)):
    if i == 6:
        yt = df_auto_train.iloc[i-6]  - 0.708 * df_auto_train.iloc[i-6]
        y_hat_t_1.append(yt)
    elif i == 7:
        yt = df_auto_train.iloc[i - 6] - 0.708 * (df_auto_train.iloc[i - 6]- y_hat_t_1[i-7]) - 0.129* df_auto_train.iloc[i-7]
        y_hat_t_1.append(yt)
    else:
        yt = df_auto_train.iloc[i - 6] - 0.708 * (df_auto_train.iloc[i - 6]- y_hat_t_1[i-7]) - \
             0.129* (df_auto_train.iloc[i-7] - y_hat_t_1[i-8])
        y_hat_t_1.append(yt)

residual1 = y_hat_t_1 - df_auto_train.iloc[6:].values

Q = Q_value(residual1,df_auto_train.iloc[6:])
chi_2_test(Q,7,8)
zero_pole_cancellation(seta1_1,0)
print(f'the variance of residual is {np.var(residual1)}.')
# ACF_PACF_Plot(residual1,20)



# y_hat_t_2 = []
# for i in range(6, len(df_auto_train)):
#     if i == 6:
#         yt = df_auto_train.iloc[i - 6] - 0.708 * df_auto_train.iloc[i - 6]
#         y_hat_t_2.append(yt)
#
#     else:
#         yt = df_auto_train.iloc[i - 6] - 0.708 * (df_auto_train.iloc[i - 6] - y_hat_t_1[i - 7])
#         y_hat_t_2.append(yt)
#
#
# residual2 = y_hat_t_2 - df_auto_train.iloc[6:].values
#
# Q = Q_value(residual2,df_auto_train.iloc[6:])
# chi_2_test(Q,0,8)
# print(f'the variance of residual is {np.var(residual1)}.')
# ACF_PACF_Plot(residual1,20)



# h step prediction
y_hat_h = []
for h in range(len(df_auto_test)):
    if h < 6:
        yh = df_auto_train.iloc[h-6] -0.747*(df_auto_train.iloc[h-6]-y_hat_t_1[h-6])-0.129* (df_auto_train.iloc[i-7] - y_hat_t_1[i-7])
        y_hat_h.append(yh)
    else:
        yh = y_hat_h[h-6]
        y_hat_h.append(yh)


forecast_error = df_auto_test - y_hat_h
print(f'the variance of forecast error is {np.var(forecast_error)}.')



# Is the derived model biased or this is an unbiased estimator?
# the estimated covariance of the estimated parameters.
cov = np.cov(seta1_1)

# plot
plt.figure()
df_auto_test.plot(label='test data')
plt.plot(df_auto_test.index,y_hat_h,label='forecast data')
plt.legend()
plt.show()

plt.figure()
df_auto_train.plot(label='train data')
plt.plot(df_auto_train.iloc[6:].index,y_hat_t_1,label='predict data')
plt.legend()
plt.show()



print('End')