# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pystan
from tqdm import tqdm

# COMMAND ----------

data = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})

# COMMAND ----------

from pylab import rcParams

rcParams['figure.figsize'] = 20, 10

Yards = data[data.NflId == data.NflIdRusher]['Yards'].values
dists = [stats.norm, stats.lognorm, stats.beta, stats.cauchy, stats.gamma,
         stats.nct, stats.t, stats.f, stats.exponnorm]
d = np.diff(np.unique(Yards)).min()

# COMMAND ----------

plt.hist(Yards, np.arange(Yards.min() - float(d) / 2, Yards.max() + float(d) / 2 + d, d), density=True, alpha=0.5, cumulative=False)
plt.xlim(-20, 40)
xt = plt.xticks()[0]
lnspc = np.linspace(min(xt), max(xt), 500)
for dist in dists:
    params = dist.fit(Yards, loc=-15)
    print(dist.name, params)
    pdf = dist.pdf(lnspc, *params)
    plt.plot(lnspc, pdf, label=dist.name)
plt.legend()
plt.show()

# COMMAND ----------

plt.hist(Yards, np.arange(Yards.min() - float(d) / 2, Yards.max() + float(d) / 2 + d, d), density=True, alpha=0.5, cumulative=True)
plt.xlim(-20, 40)
xt = plt.xticks()[0]
lnspc = np.linspace(min(xt), max(xt), 500)
for dist in dists:
    params = dist.fit(Yards, loc=-15)
    print(dist.name, params)
    pdf = dist.cdf(lnspc, *params)
    plt.plot(lnspc, pdf, label=dist.name)

plt.legend()
plt.show()

# COMMAND ----------

model_code = """
data {
    int N_samples;
    vector[N_samples] Yards;
    matrix[N_samples, 2] X;
    int N_Rushers;
    int<lower=1, upper=N_Rushers> IdRusher[N_samples];
}

parameters {
    real mu;
    real<lower=0> sigma;
    vector[N_Rushers] lam;
    real mu_lam;
    real<lower=0> sigma_lam;
    vector[2] beta_lam;
}

model {
    lam ~ normal(mu_lam, sigma_lam);
    Yards ~ exp_mod_normal(mu, sigma, exp(lam[IdRusher] + X * beta_lam));
}
"""
%time model = pystan.StanModel(model_code=model_code)

# COMMAND ----------

NflIdRusher_dict = {v: k + 1 for k, v in data['NflIdRusher'].drop_duplicates().sort_values().reset_index(drop=True).to_dict().items()}


def map_NflIdRusher(NflIdRusher):
    return NflIdRusher_dict.get(NflIdRusher, 0)


def make_standata(data):
    data = data[data['NflId'] == data['NflIdRusher']].copy()
    data['IdRusher'] = data['NflIdRusher'].map(map_NflIdRusher)
    data['newYardLine'] = data.apply(lambda row: row['YardLine'] if row['PossessionTeam']==row['FieldPosition'] else 100-row['YardLine'], axis=1)

    standata = {
        'N_samples': len(data),
        'Yards': data.get('Yards', pd.DataFrame()).values,
        'X': data[['A', 'Distance']].values,
        'N_Rushers': max(data['IdRusher']),
        'IdRusher': data['IdRusher'].values,
        'newYardLine': data['newYardLine'].values}
    return (standata)

# COMMAND ----------

standata = make_standata(data)


def init():
    return dict(lam=np.zeros(standata['N_Rushers']))

%time fit = model.sampling(data=standata, iter=1000, chains=4, thin=1, init=init, algorithm="NUTS", seed=0)

# COMMAND ----------

print(fit)

# COMMAND ----------

fit.plot()
plt.show()

# COMMAND ----------

from kaggle.competitions import nflrush
env = nflrush.make_env()

# COMMAND ----------

parameters = fit.extract(permuted=True)
sigma = parameters['sigma']
mu = parameters['mu']
lam = np.column_stack([parameters['mu_lam'], parameters['lam']])
beta = parameters['beta_lam']
for test_df, sample_df in tqdm(env.iter_test()):
    standata = make_standata(test_df)
    K = 1 / (np.exp(lam[:, standata['IdRusher']] + np.dot(beta, standata['X'].T)).reshape(-1) * sigma)
    pred = [np.mean(stats.exponnorm.cdf(x=x, K=K, loc=mu, scale=sigma)) for x in range(-99, 100)]
    preds_df = pd.DataFrame(data=[pred], columns=sample_df.columns)
    preds_df.loc[:, ('Yards' + str(int(100 - standata['newYardLine'][0]))):'Yards99'] = 1
    preds_df.loc[:, 'Yards-99':'Yards' + str(max(int(-1 - standata['newYardLine'][0]), -99))] = 0
    env.predict(preds_df)
env.write_submission_file()
