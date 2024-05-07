
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from prophet import Prophet
from neuralprophet import NeuralProphet
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import shapiro
from scipy.stats import boxcox

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf


os.environ['CMDSTAN'] = ""


def get_lags(df):
    pacf_values, conf_intervals = pacf(df['norm_resid'].squeeze(), nlags=5, alpha=0.05)

    # Get the PACF values and significance intervals
    significant_lags = []
    # Calculate significance boundaries (blue region)
    lower_bound = -1.96 / np.sqrt(len(df))  # 95% significance level for two-tailed test
    upper_bound = 1.96 / np.sqrt(len(df))


    # Find lags with PACF values outside the 95% CI
    # Get the lags whose PACF values fall outside the confidence interval
    # significant_lags = np.where((pacf_values > conf_intervals[:, 1]) | (pacf_values < conf_intervals[:, 0]))[0]
    significant_lags = np.where((pacf_values > upper_bound) | (pacf_values < lower_bound))[0]
    

    return significant_lags, pacf_values


def seasonal_de_day(df,key,plot):
    if len(df)<28:
        return [], pd.DataFrame()
    dt_range = pd.to_datetime(pd.date_range(start='07/01/2021', end='10/31/2021')).date
    mis = [i for i in dt_range if i not in df.index]
    df = pd.concat([df,pd.DataFrame(index=mis)])
    df = df.sort_index()
    df['ds'] = df.index
    
    # print(df.tail())
    df = df.rename(columns={key:'y'})
    df['y'] = np.log1p(df['y'])
    
    holiday = pd.DataFrame({
      'holiday': 'Yom Kippur',#Jew holiday, more than 10% residents are Jew in NYC
      'ds': pd.to_datetime(['2021-07-04','2021-07-05','2021-09-06','2021-09-16','2021-10-11']),
      'lower_window': 0,
      'upper_window': 1,
    })

    fb = Prophet(interval_width= 0.80,weekly_seasonality=True,holidays=holiday)#%uncertainty interval

    fb.add_country_holidays(country_name='US')

    fb.fit(df)

    forecast = fb.predict(df)

    
    
    if plot:
        fb.plot(forecast)
        plt.savefig(r'forecast.png',format='png',bbox_inches='tight',dpi=300)
        plt.show()
        plt.close('all')
        fb.plot_components(forecast) 
        plt.savefig(r'components.png',format='png',bbox_inches='tight',dpi=300)
        plt.show()
        plt.close('all')
    

    forecast = forecast.set_index('ds')
    forecast['y'] = df['y']

    forecast['resid'] = (df['y']-forecast['yhat'])

    forecast['abs_resid'] = abs(df['y']-forecast['yhat'])
    forecast['dayofweek'] =forecast.index.dayofweek
    norm = forecast.groupby('dayofweek').mean()
    norm = norm.rename(columns={'abs_resid':'avg_abs_resid'})
    forecast = forecast.reset_index()
    forecast = forecast.merge(norm['avg_abs_resid'], on='dayofweek',how='outer')

    forecast['norm_resid'] = (forecast['resid'])/(forecast['avg_abs_resid'])
    
    
    forecast['date']  = pd.to_datetime(forecast.ds.dt.date)
    
    
    forecast = forecast.set_index('ds')

    forecast.loc[forecast['date'].isin(holiday['ds']),'norm_resid'] = np.mean(abs(forecast['norm_resid']))

    resid = (forecast['norm_resid'].dropna()) 
    
    
    skew = resid.skew()
    if -1< skew < -0.5:#中度负偏态
        resid = np.sqrt(max(resid)+1-resid)
    elif 1 > skew > 0.5: #中度正偏态
        if min(resid)<0:
            resid = resid-min(resid)
        resid = np.sqrt(resid)
    elif skew >=1:#高度正偏态
        if min(resid)<0:
            resid = resid-min(resid)
        resid = np.log1p(resid)
    elif skew<=-1:#高度负偏态
        resid = np.log1p(max(resid)+1-resid)
    
    
    zscore = stats.zscore(resid.dropna())
    abs_zscore = abs(zscore)
    if(len(zscore)==0):
        print(resid.dropna())
        print(forecast['norm_resid'])
    forecast['zscore'] = zscore
    forecast['abs_zscore'] = abs_zscore
    var = forecast['norm_resid'].fillna(0).var()
    
    lags,rhos = get_lags(forecast)
    if len(lags)>1:
        rhos_sum = 0
        for lagi in lags:
            if lagi==0:
                continue
            rho = rhos[lagi]
            rhos_sum+=rho**2

        var_new = var/(1-rhos_sum)
        abs_ZS_new =abs(forecast['norm_resid']-forecast['norm_resid'].mean())/np.sqrt(var_new)
        forecast['abs_ZS_new'] = abs_ZS_new
    else:
        abs_ZS_new = abs(forecast['norm_resid']-forecast['norm_resid'].mean())/np.sqrt(var)
        forecast['abs_ZS_new'] = abs_ZS_new
        

    forecast.reset_index(inplace=True)
    
    outliers = df.loc[abs_ZS_new[abs_ZS_new > 3].index].index
    
    return outliers,forecast
    
    
def seasonal_de_hour(df,key,plot):
    # df = df[df[key]>24]
    if len(df)<28*6:
        return [], pd.DataFrame()
    start = pd.to_datetime('07/01/2021 00:00:00')
    end = pd.to_datetime('10/31/2021 23:59:59')
    dt_range = pd.to_datetime(pd.date_range(start=start, end=end,freq='4H'))
    mis = [i for i in dt_range if i not in df.index]
    df = pd.concat([df,pd.DataFrame(index=mis)])
    df = df.sort_index()

    df['ds'] = df.index
    df = df.rename(columns={key:'y'})
    df['y'] = np.log1p(df['y'])
    # print(df)
    holiday = pd.DataFrame({
      'holiday': 'Yom Kippur',#Jew holiday, more than 10% residents are Jew in NYC
      'ds': pd.to_datetime(['2021-07-04','2021-07-05','2021-09-06','2021-09-16','2021-10-11']),
      'lower_window': 0,
      'upper_window': 1,
    })
    
    fb = Prophet(interval_width= 0.80,daily_seasonality=True,weekly_seasonality=True,holidays=holiday)
    
    fb.fit(df)
    forecast = fb.predict(df)

    
    if plot:
        fb.plot(forecast)
        plt.show()
        plt.close('all')
        fb.plot_components(forecast) 
        plt.show()
        plt.close('all')
        
    
    
    daily_totals = df.groupby(df['ds'].dt.date)['y'].sum()
    daily_totals_df = pd.DataFrame({'date': daily_totals.index, 'daily_total': daily_totals.values})
    df = df.merge(daily_totals_df, left_on=df['ds'].dt.date, right_on=daily_totals_df['date'], how='left')
    df['daily_total'] = df['daily_total'].ffill()

    
    df = df.set_index('ds')

    
    forecast = forecast.set_index('ds')
    forecast['y'] = df['y']

    forecast['resid'] = (df['y']-forecast['yhat'])#/df['y']

    forecast['abs_resid'] = abs(df['y']-forecast['yhat'])
    forecast['dayofweek'] =forecast.index.dayofweek
    
    forecast['hour'] = forecast.index.hour
    norm = forecast.groupby(['dayofweek','hour']).mean()
    norm = norm.rename(columns={'abs_resid':'avg_abs_resid'})
    
    forecast = forecast.reset_index()
    forecast = forecast.merge(norm['avg_abs_resid'], on=['dayofweek','hour'],how='outer')
    

    forecast['norm_resid'] = forecast['resid']/(forecast['avg_abs_resid'])
    
    forecast['date']  = pd.to_datetime(forecast.ds.dt.date)
    forecast.loc[forecast['date'].isin(holiday['ds']),'norm_resid'] = np.mean(abs(forecast['norm_resid']))
    
    forecast = forecast.set_index('ds')
    
    
    resid = (forecast['norm_resid'].dropna())
    
    skew = resid.skew()
    if -1< skew < -0.5:#中度负偏态
        resid = np.sqrt(max(resid)+1-resid)
    elif 1 > skew > 0.5: #中度正偏态
        if min(resid)<0:
            resid = resid-min(resid)
        resid = np.sqrt(resid)
    elif skew >=1:
        if min(resid)<0:
            resid = resid-min(resid)
        resid = np.log1p(resid)
    elif skew<=-1:
        resid = np.log1p(max(resid)+1-resid)
    

    
    zscore = stats.zscore(resid.dropna())
       

    abs_zscore = abs(zscore)
    
    forecast['zscore'] = zscore
    forecast['abs_zscore'] = abs_zscore

    var = forecast['norm_resid'].fillna(0).var()
    
    lags,rhos = get_lags(forecast)
    if len(lags)>1:
        rhos_sum = 0
        for lagi in lags:
            if lagi==0:
                continue
            rho = rhos[lagi]
            rhos_sum+=rho**2
        
        var_new = var/(1-rhos_sum)
        if var_new <0:
            print(var_new)
        abs_ZS_new =abs(forecast['norm_resid']-forecast['norm_resid'].mean())/np.sqrt(var_new)
        forecast['abs_ZS_new'] = abs_ZS_new
    else:
        abs_ZS_new = abs(forecast['norm_resid']-forecast['norm_resid'].mean())/np.sqrt(var)
        forecast['abs_ZS_new'] = abs_ZS_new

    forecast.reset_index(inplace=True)
    
    

    outliers = df.loc[abs_ZS_new[abs_ZS_new > 3].index].index
    
    return outliers,forecast
    