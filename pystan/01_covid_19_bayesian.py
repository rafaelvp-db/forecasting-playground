# Databricks notebook source
# MAGIC %md
# MAGIC * For the predictive analytics of coronavirus spread, we used a logistic curve model.
# MAGIC * Bayesian inference makes it possible to obtain probability density functions for model parameters and estimate the uncertainty that is important in the risk assessment analytics.
# MAGIC * In Bayesian regression approach, we can take into account expert opinions via information prior distribution. For Bayesian inference calculations, we used python pystan package.
# MAGIC * New historical data will correct the distributions for model parameters and forecasting results.
# MAGIC * In the practical analytics, it is important to find the maximum of coronavirus cases per day, this point means estimated half time of coronavirus spread in the region under investigation.
# MAGIC 
# MAGIC More details of our study are [here](http://www.linkedin.com/pulse/using-logistic-curve-bayesian-inference-modeling-bohdan-pavlyshenko/).

# COMMAND ----------

!unzip /dbfs/FileStore/rafael.pierre/covid19_global_forecasting_week_2.zip

# COMMAND ----------

import pystan
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns 
%matplotlib inline
import pystan
import datetime
sns.set()

# Set up options 

# File name of historical data file
data_filename='/dbfs/FileStore/rafael.pierre/covid/train.csv'

# Name of fields in the hystorical data frame for in the following order: 
# date, region, confirmed cases, fatalities
field_names=['Date','Country_Region','ConfirmedCases','Fatalities']

# List of regions for prediction
region_list=['China','US','Italy','Spain','Iran','Germany','France']

# Number of days for prediction
n_days_predict=25

# fields names:
region_field, cases_field,fatalities_field='region','cases','deaths'

# Normalization coefficients
target_field_norm_coef=1/100000
time_var_norm_coef=1/7

# fields names:
region_field, cases_field,fatalities_field='region','cases','deaths'


# COMMAND ----------

model_logistic = """
    data {
        int<lower=1> n;
        int<lower=1> n_pred;
        vector[n] y;
        vector[n] t;
        vector[n_pred] t_pred;
    }
    parameters {
        real<lower=0> alpha;
        real<lower=0> beta;
        real<lower=0> t0;
        real<lower=0> sigma; 
    }
    model {
    alpha~normal(1,1);
    beta~normal(1,1);
    t0~normal(10,10);
    y ~ normal(alpha ./ (1 + exp(-(beta*(t-t0)))), sigma);
    }
    generated quantities {
      vector[n_pred] pred;
      for (i in 1:n_pred)
      pred[i] = normal_rng(alpha / (1 + exp(-(beta*(t_pred[i]-t0)))),sigma);
    }
    """
    
def plot_results(df_res,target_field,region_value, fit_samples):
    fig, ax = plt.subplots(2,2, sharex=False, figsize=(15,10))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    df_res['pred'].plot(yerr=df_res['pred_yerr'].values,title='Prediction ($10^5$)',ecolor='#aaaaaa',ax=ax[0,0])
    ax[0,0].scatter(df_res.index,df_res.y,s=10,c='green')
    df_res[['n_per_day_real','n_per_day_prediction']]=df_res[['y','pred']]-df_res[['y','pred']].shift(1)
    df_res[['n_per_day_real','n_per_day_prediction']].plot(ax = ax[0,1],title='Number per day ($10^5$)')
    alpha_mean=np.round(fit_samples['alpha'].mean(),3)
    alpha_std=np.round(fit_samples['alpha'].std(),3)
    beta_mean=np.round(fit_samples['beta'].mean(),3)
    beta_std=np.round(fit_samples['beta'].std(),3)
    t0_mean=np.round(fit_samples['t0'].mean(),3)
    t0_std=np.round(fit_samples['t0'].std(),3)
    print (f'Model parameters: alpha={alpha_mean} (sd:{alpha_std}), beta={beta_mean}(sd:{beta_std}),\
    t0={t0_mean}(sd:{t0_std})')
    alpha_samples=np.round(pd.Series(fit_samples['alpha']),3)
    alpha_samples.plot(kind='density', 
    title=r'$\alpha$'+f' (mean:{alpha_mean}, sd:{alpha_std})',ax = ax[1,0])
    beta_samples=pd.Series(fit_samples['beta'])
    beta_samples.plot(kind='density', 
    title=r'$\beta$'+f' (mean:{beta_mean}, sd:{beta_std})',ax = ax[1,1])
    fig.suptitle(f'Number of {target_field} for {region_value}')
    plt.show()
    
def get_prediction(df,stan_model, n_days_predict=25, target_field='cases',
                   region_field='region', region_value='China',
                   target_field_norm_coef=target_field_norm_coef,
                   time_var_norm_coef=time_var_norm_coef):
    df.date=pd.to_datetime(df.date)
    df_res=df.loc[df[region_field]==region_value, ['date',target_field]].set_index('date')\
    .groupby(pd.Grouper(freq='D'))\
    [target_field].sum().to_frame('y').reset_index()
    print ('Time Series size:',df_res.shape[0])
    n_train=df_res.shape[0]
    maxdate=df_res.date.max()
    for i in np.arange(1,n_days_predict+1):
        df_res=df_res.append(pd.DataFrame({'date':\
            [maxdate+datetime.timedelta(days=int(i))]}))
    df_res['t']=time_var_norm_coef*np.arange(df_res.shape[0])
    df_res.y=target_field_norm_coef*df_res.y
    df_res.set_index('date',inplace=True)
    data = {'n': n_train,'n_pred':df_res.shape[0],
            'y': df_res.iloc[:n_train,:].y.values,'t':df_res.iloc[:n_train,:]\
            .t.values,'t_pred':df_res.t.values}
    fit=stan_model.sampling(data=data, iter=5000, chains=3)
    fit_samples = fit.extract(permuted=True)
    pred=fit_samples['pred']
    df_res['pred']=pred.mean(axis=0)
    df_res['pred_yerr']=(pd.DataFrame(pred).quantile(q=0.95,axis=0).values-\
                       pd.DataFrame(pred).quantile(q=0.05,axis=0).values)/2
    plot_results(df_res,target_field,region_value,fit_samples)
    return(df_res)
    
def get_regions_prediction(df,stan_model, region_list,n_days_predict=25, 
    region_field=region_field,target_field='cases'):
    for i in region_list:
        print (f'\n{i}:')
        df_res=get_prediction(df,stan_model, n_days_predict=n_days_predict, target_field=target_field, 
        region_field=region_field,region_value=i)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compile model

# COMMAND ----------

stan_model= pystan.StanModel(model_code=model_logistic)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read data

# COMMAND ----------

!unzip /dbfs/FileStore/rafael.pierre/covid19_global_forecasting_week_2.zip -d /dbfs/FileStore/rafael.pierre/covid

# COMMAND ----------

df=pd.read_csv(data_filename)[field_names]
df.columns=['date',region_field,cases_field,fatalities_field]
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get prediction for one region

# COMMAND ----------

df_res=get_prediction(df,stan_model,n_days_predict=25, target_field='cases', region_value='China')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get prediction for list of regions

# COMMAND ----------

print ('\nNumber of cases for regions:')
get_regions_prediction(df,stan_model, region_list,target_field='cases')
