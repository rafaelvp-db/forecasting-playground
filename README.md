## Forecasting Playground

**TLDR;** multiple examples of causal and probabilistic time series forecasting notebooks for [Databricks](https://www.databricks.com)

### Example: PyMC

<img src="https://github.com/rafaelvp-db/forecasting-playground/raw/master/img/posterior_covid.png" />

### Tools & Frameworks

* [PyStan](https://pystan.readthedocs.io/en/latest/)
* [PyMC](https://www.pymc.io/welcome.html)
* [Prophet](https://facebook.github.io/prophet/)
* [Time Series Transformer](https://huggingface.co/docs/transformers/model_doc/time_series_transformer)

### Instructions

* Clone this repo into your Databricks Workspace
* Create a cluster with Machine Learning Runtime version 12.0
* Run the example notebooks

### Datasets Used

* [COVID-19](https://www.kaggle.com/datasets/imdevskp/corona-virus-report)
* [NFL Big Data Bowl](https://www.kaggle.com/c/nfl-big-data-bowl-2022)
* [COVID-19 Deaths Registered Monthly in England and Wales](https://github.com/pymc-devs/pymc-examples/blob/2022.12.0/examples/data/deaths_and_temps_england_wales.csv)
* [Monash Time Series Forecasting Dataset](https://huggingface.co/datasets/monash_tsf)

### Reference

* [Bayesian Model for COVID-19](https://www.kaggle.com/code/bpavlyshenko/bayesian-model-for-covid-19-spread-prediction/log)
* [Hierarchical Bayesian Modeling using PyStan](https://www.kaggle.com/code/kyonaganuma/hierarchical-bayesian-modeling-using-pystan)
* [Probabilistic Time Series Forecasting with Transformers](https://huggingface.co/blog/time-series-transformers)
* [Counterfactual inference: calculating excess deaths due to COVID-19](https://github.com/pymc-devs/pymc-examples/blob/2022.12.0/examples/causal_inference/excess_deaths.ipynb)
