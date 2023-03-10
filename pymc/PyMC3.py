# Databricks notebook source
# MAGIC %md
# MAGIC ## Introduction to PyMC3
# MAGIC 
# MAGIC ### Prerequisite - Course 2
# MAGIC 
# MAGIC ### Attribution
# MAGIC 
# MAGIC It is important to acknowledge the authors who have put together fantastic resources that have allowed me to make this notebook possible. 
# MAGIC 
# MAGIC 
# MAGIC 1. *The majority of the the examples here are taken from the book 'Introduction to statistical modeling and probabilistic programming using PyMC3 and ArviZ', Second Edition by Osvaldo Martin*
# MAGIC 
# MAGIC 2. [PyMC3 website](docs.pymc.io)
# MAGIC 
# MAGIC 3. Bayesian Methods for Hackers by Davidson-Pilon Cameron
# MAGIC 
# MAGIC 4. Doing Bayesian Data Analysis by John Kruschke
# MAGIC 
# MAGIC ### Overview of Probabilistic Programming 
# MAGIC 
# MAGIC An overview of probabilistic frameworks is given in this [post](https://eigenfoo.xyz/prob-prog-frameworks/)  by George Ho, one of the developers of PyMC3. He outlines the components needed for a probabilistic framework in this figure 
# MAGIC 
# MAGIC <img src="https://eigenfoo.xyz/assets/images/prob-prog-flowchart.png" width="400">
# MAGIC 
# MAGIC <br></br>
# MAGIC <center>Figure from George Ho's post 'Anatomy of a Probabilistic Programming Framework"</center>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### What is PyMC3?
# MAGIC 
# MAGIC PyMC3 is a probabilistic programming framework for performing Bayesian modeling and visualization. It uses Theano as a backend. It has algorithms to perform Monte Carlo simulation as well as Variational Inference. It also has a diagnostic visualization tool called ArViz.
# MAGIC 
# MAGIC It can be used to infer values of parameters of models that we are unsure about by utilizing the observed data. A good example is given here [https://docs.pymc.io/notebooks/ODE_API_introduction.html](https://docs.pymc.io/notebooks/ODE_API_introduction.html). 
# MAGIC 
# MAGIC $$ \dfrac{dy}{dt} = m g - \gamma y$$
# MAGIC We are trying to estimate the parameters of air resistance (\\(\gamma\\)) from the Ordinary Differential Equation (ODE) of freefall. We have an understanding of the physics behind freefall as represented by the ODE and we have observed/measured some of the variables such as mass (m), position (y) and velocity (\\(\dfrac{dy}{dt}\\)) but we don't know what the parameter of air resistance is here. We can use PyMC3 to perform inference and give us a distribution of potential values of air resistance. A key point to note here is that the more information we have regarding other variables, the more certainty we have in our desired variable (air resistance). Suppose we are unsure about the gravitational constant (g) used in the ODE (implemented by specifying a prior distribution as opposed to a constant value of 9.8), we get more uncertainty in the air resistance variable as well.
# MAGIC 
# MAGIC 
# MAGIC ### General Structure of PyMC3
# MAGIC 
# MAGIC It consists of phenomena represented by equations made up of random variables and deterministic variables. The random variables can be divided into observed variables and unobserved variables. The observed variables are those for which we have data and the unobserved variables are those for which we have to specify a prior distribution.
# MAGIC 
# MAGIC #### Observed Variables
# MAGIC 
# MAGIC ```
# MAGIC with pm.Model():
# MAGIC     obs = pm.Normal('x', mu=0, sd=1, observed=np.random.randn(100))
# MAGIC ```
# MAGIC 
# MAGIC #### Unobserved Variables
# MAGIC 
# MAGIC ```
# MAGIC with pm.Model():
# MAGIC     x = pm.Normal('x', mu=0, sd=1)
# MAGIC ```
# MAGIC 
# MAGIC We will look at an example of Linear Regression to illustrate the fundamental features of PyMC3.
# MAGIC 
# MAGIC ### An example with Linear Regression
# MAGIC 
# MAGIC The example below illustrates linear regression with a single output variable and two input variables.
# MAGIC 
# MAGIC $$ y = \alpha + \beta_1 x_1 +  \beta_2 x_2 + \sigma_{error} $$

# COMMAND ----------

!pip install -q --upgrade pip && pip install pymc3 graphviz

# COMMAND ----------

import pymc3
pymc3.__version__

# COMMAND ----------

# MAGIC %md
# MAGIC #### Generate the Data

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import arviz as az
# MAGIC import numpy as np
# MAGIC import warnings
# MAGIC warnings.filterwarnings("ignore")
# MAGIC import matplotlib.pyplot as plt
# MAGIC import graphviz
# MAGIC import os
# MAGIC 
# MAGIC os.environ['OMP_NUM_THREADS'] = '4'
# MAGIC 
# MAGIC # Initialize random number generator
# MAGIC np.random.seed(123)
# MAGIC 
# MAGIC # True parameter values
# MAGIC alpha, sigma = 1, 1
# MAGIC beta = [1, 2.5]
# MAGIC 
# MAGIC # Size of dataset
# MAGIC size = 100
# MAGIC 
# MAGIC # Predictor variable
# MAGIC X1 = np.linspace(0, 1, size)
# MAGIC X2 = np.linspace(0,.2, size)
# MAGIC 
# MAGIC # Simulate outcome variable
# MAGIC Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

# COMMAND ----------

import pymc3 as pm
from pymc3 import Model, Normal, HalfNormal
from pymc3 import find_MAP

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Setup in PyMC3

# COMMAND ----------

basic_model = Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = Normal('alpha', mu=0, sd=5)
    beta = Normal('beta', mu=0, sd=5, shape=2)
    sigma = HalfNormal('sigma', sd=4)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2 
    
    # Deterministic variable, to have PyMC3 store mu as a value in the trace use
    # mu = pm.Deterministic('mu', alpha + beta[0]*X1 + beta[1]*X2)
    
    # Likelihood (sampling distribution) of observations
    Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

pm.model_to_graphviz(basic_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Plate Notation
# MAGIC 
# MAGIC A way to graphically represent variables and their interactions in a probabilistic framework.
# MAGIC 
# MAGIC [Wikipedia Plate Notation](https://en.wikipedia.org/wiki/Plate_notation)
# MAGIC 
# MAGIC #### MAP Estimate 
# MAGIC 
# MAGIC PyMC3 computes the MAP estimate using numerical optimization, by default using the BFGS algorithm. These provide a point estimate which may not be accurate if the mode does not appropriately represent the distribution.

# COMMAND ----------

map_estimate = find_MAP(model=basic_model, maxeval=10000)
map_estimate

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inference in PyMC3

# COMMAND ----------

from pymc3 import NUTS, sample
from scipy import optimize

# COMMAND ----------

with basic_model:
    
    trace = sample(3000, model = basic_model)

# COMMAND ----------

# MAGIC %md
# MAGIC You can also pass a parameter to step that indicates the type of sampling algorithm to use such as
# MAGIC 
# MAGIC * Metropolis
# MAGIC * Slice sampling
# MAGIC * NUTS
# MAGIC 
# MAGIC PyMC3 can automatically determine the most appropriate algorithm to use here, so it is best to use the default option.
# MAGIC 
# MAGIC #### Distribution Information through Traceplots

# COMMAND ----------

trace['alpha']

# COMMAND ----------

from pymc3 import traceplot

# COMMAND ----------

traceplot(trace)

# COMMAND ----------

# MAGIC %md
# MAGIC We will look at the summary of the sampling process. The columns will be explained as we progress through this course.

# COMMAND ----------

az.summary(trace)

# COMMAND ----------

from pymc3 import summary
summary(trace)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Composition of Distributions for Uncertainty
# MAGIC 
# MAGIC You can do the same without observations, to perform computations and get uncertainty quantification. In this example add two normally distributed variables to get another normally distributed variable. By definition
# MAGIC 
# MAGIC $$\mu_c = \mu_a + \mu_b $$
# MAGIC 
# MAGIC $$\sigma_c^2 = \sigma_a^2 + \sigma_b^2$$

# COMMAND ----------

basic_model2 = Model()

with basic_model2:

    # Priors for unknown model parameters
    alpha = Normal('alpha', mu=20, sd=2)
    beta = Normal('beta', mu=10, sd=2, shape=1)
    # These aren't used in the calculation, but please experiment by composing various combinations
    # of these function for calculating mu
    sigma = HalfNormal('sigma', sd=1)
    ??_obs = pm.HalfCauchy("??_obs", beta=1, testval=0.1)

    # Expected value of outcome
    mu = pm.Deterministic('mu', alpha + beta) 
    
    trace = sample(1000)
    
print(trace['mu'])
traceplot(trace)    

# COMMAND ----------

# Note the mean and standard deviation of the variable mu. We didn't need observations to compute this uncertainty.
try:
    # older syntax
    pm.plot_posterior(trace, var_names=['alpha', 'beta', 'mu'], credible_interval=0.68)
except:
    pm.plot_posterior(trace, var_names=['alpha', 'beta', 'mu'], hdi_prob=0.68)

# COMMAND ----------

# MAGIC %md
# MAGIC #### HPD, Credible Interval, HDI and ROPE
# MAGIC 
# MAGIC ##### What is it used for?
# MAGIC 
# MAGIC HDI, HPD and ROPE are essentially used for making decisions from the posterior distribution.
# MAGIC 
# MAGIC ##### HPD and Credible Interval
# MAGIC 
# MAGIC For example, if we plot the posterior of a beta distribution with some parameters, the credible interval for the Highest Posterior Density (HPD) is the **shortest** interval that has the given probability indicated by the HPD. This interval is also called the HPD interval. As the name indicates, this involves regions of the highest posterior probability density. For unimodal distributions, this includes the mode.
# MAGIC 
# MAGIC 
# MAGIC ![HPD](https://courseraimages.s3-us-west-2.amazonaws.com/HPDI_1.png)
# MAGIC 
# MAGIC x% HPD interval
# MAGIC 
# MAGIC ![HPD](https://courseraimages.s3-us-west-2.amazonaws.com/HPDI_2.png)
# MAGIC 
# MAGIC x% HPD interval
# MAGIC 
# MAGIC 
# MAGIC ##### HDI
# MAGIC 
# MAGIC A related term is the Highest Density Interval (HDI) which is a more general term that can apply for any distribution such as a prior and not just the posterior. In other words a posterior's HDI is called the HPD interval. 
# MAGIC 
# MAGIC As an example, if we suspect that the dice used at a casino is loaded, we can infer the probability of getting the value 3 from the six possible outcomes. Ideally, this should be 1/6 = 0.16666. If this happens to fall in the HPD interval, we can assume that the dice is fair however it may be that the distribution may be biased to one side or the other.  
# MAGIC 
# MAGIC 
# MAGIC ##### ROPE
# MAGIC 
# MAGIC What is the probability of getting a value given by x? We can't really calculate this exactly but we can compute this probability within a range given by x + $\Delta$x, x - $\Delta$x. 
# MAGIC 
# MAGIC ![HPD](https://courseraimages.s3-us-west-2.amazonaws.com/HPD.png)
# MAGIC 
# MAGIC <i>Probability of getting values less than x, and a range of values around x</i>
# MAGIC 
# MAGIC 
# MAGIC Sometimes, instead of looking at the probability that x = 0.16666, we look at the probability that it falls within the range 0.12 and 0.20. This range is called the Region of Practical Equivalence or ROPE. This implies that, based on our subjective opinion, getting a value between 0.12 and 0.20 is practically equivalent to getting a 0.16666. Hence, we can assume that the dice is fair given any value within this range. ROPE allows us to make decisions about an event from an inferred posterior distribution. After computing the posterior, the ROPE given by 0.12 and 0.20 can either overlap with the HPD (of getting a 3)  
# MAGIC 
# MAGIC * completely
# MAGIC * not overlap at all 
# MAGIC * partially overlap with the HPD
# MAGIC 
# MAGIC Complete overlap suggests that our computed probability coincides with what we would expect from a fair dice. If it does not overlap, it is not a fair dice and a partial overlap indicates that we cannot be certain that is either fair or unfair.
# MAGIC 
# MAGIC In short, we define a ROPE based on our subject matter expertise and compare it to the HPD to make a decision from the posterior distribution.
# MAGIC 
# MAGIC ##### Credible intervals vs. Confidence Intervals
# MAGIC 
# MAGIC This deserves special mention particularly due to the subtle differences stemming from the Bayesian (credible intervals) vs. Frequentist (confidence intervals) approaches involved. Bayesians consider the parameters to be a distribution, and for them there is no true parameter. However, Frequentists fundamentally assume that there exists a true parameter. 
# MAGIC 
# MAGIC * Confidence intervals quantify our confidence that the true parameter exists in this interval. It is a statement about the interval.
# MAGIC * Credible intervals quantify our uncertainty about the parameters since there are no true parameters in a Bayesian setting. It is a statement about the probability of the parameter.
# MAGIC 
# MAGIC For e.g. if we are trying to estimate the R0 for COVID-19, one could say that we have a 95% confidence interval of the true R0 being between 2.0 and 3.2. In a Bayesian setting, the 95% credible interval of (2, 3.2) implies that 95% of the possible R0 values fall between 2.0 and 3.2.
# MAGIC 
# MAGIC We will see how we can visualize the following using ArViz
# MAGIC 
# MAGIC * HDI (Black lines)
# MAGIC * ROPE (Green lines)

# COMMAND ----------

import numpy as np
from scipy import stats as stats 

np.random.seed(1)

try:
    az.plot_posterior({'??':stats.beta.rvs(5, 5, size=20000)},
                  credible_interval=0.75,                    # defaults to 94%
                  #hdi_prob = 0.85,  
                  rope =[0.45, 0.55])
except:                      
    az.plot_posterior({'??':stats.beta.rvs(5, 5, size=20000)},
                      #credible_interval=0.75,                    # defaults to 94%
                      hdi_prob = 0.85,                            # credible_interval is deprecated, use hdi_prob
                      rope =[0.45, 0.55])

# COMMAND ----------

# MAGIC %md
# MAGIC Another way to do this is by plotting a reference value on the posterior. Below, a reference value of 0.48 is used and it can be seen that 45.2% of the posterior is below this value while 54.8% of the posterior is above this value. If we were estimating our parameter to have a value of 0.48, this suggests a good fit but with a slight right bias or in other words that the parameter is likely to have a value greater than 0.48.

# COMMAND ----------

try:
    az.plot_posterior({'??':stats.beta.rvs(5, 5, size=20000)},
                  credible_interval=0.75,                    # defaults to 94%
                  #hdi_prob = 0.85,                            # credible_interval is deprecated, use hdi_prob
                  ref_val=0.48)
except:
    az.plot_posterior({'??':stats.beta.rvs(5, 5, size=20000)},
                      #credible_interval=0.75,                    # defaults to 94%
                      hdi_prob = 0.85,                            # credible_interval is deprecated, use hdi_prob
                      ref_val=0.48)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Modeling with a Gaussian Distribution
# MAGIC 
# MAGIC Gaussians (Normal distributions) are normally used to approximate a lot of practical data distributions. Some of the reasons for this are: 
# MAGIC 
# MAGIC * The Central Limit Theorem, which states: 
# MAGIC 
# MAGIC    ```The distribution of the sample means will be a normal distribution```
# MAGIC 
# MAGIC    [Intuitive explanation of the Central Limit Theorem](https://statisticsbyjim.com/basics/central-limit-theorem/)
# MAGIC 
# MAGIC    which implies that if we take the mean of the sample means, we should get the true population mean. 
# MAGIC 
# MAGIC * A more subtle reason for adopting the Gaussian distribution to represent a lot of phenomena is the fact that a lot of these phenomena themselves are a result of averages of varying factors. 
# MAGIC 
# MAGIC * Mathematical tractability of the distribution - it is easy to compute in closed form. While not every distribution can be approximated with a single Gaussian distribution, we can use a mixture of Gaussians to represent other multi-modal distributions.
# MAGIC 
# MAGIC The probability density for a Normal distribution in a single dimension is given by:
# MAGIC 
# MAGIC $p(x) = \dfrac{1}{\sigma \sqrt{2 \pi}} e^{-(x - \mu)^2 / 2 \sigma^2}$
# MAGIC 
# MAGIC where $\mu$ is the mean and $\sigma$ is the standard deviation. In higher dimensions, we have a vector of means and a covariance matrix. 
# MAGIC 
# MAGIC #### Example with PyMC3
# MAGIC 
# MAGIC We read the chemical shifts data, and plot the density to get an idea of 
# MAGIC the data distribution. It looks somewhat like a Gaussian so maybe we can start
# MAGIC there. We have two parameters to infer, that is the mean and the standard deviation. 
# MAGIC We can estimate a prior for the mean by looking at the density and putting 
# MAGIC some bounds using a uniform prior. The standard deviation is however chosen to 
# MAGIC have a mean-centered half-normal prior (half-normal since the standard deviation 
# MAGIC cannot be negative). We can provide a hyperparameter for this by inspecting the
# MAGIC density again. These values decide how well we converge to a solution so good 
# MAGIC values are essential for good results. 

# COMMAND ----------

data = np.loadtxt('data/chemical_shifts.csv')
az.plot_kde(data, rug=True)
plt.yticks([0], alpha=0)

# COMMAND ----------

data

# COMMAND ----------

import pymc3 as pm
from pymc3.backends import SQLite, Text
model_g = Model()

with model_g:
    
    #backend = SQLite('test.sqlite')
    db = pm.backends.Text('test')
    ?? = pm.Uniform('??', lower=40, upper=70)
    ?? = pm.HalfNormal('??', sd=10)
    y = pm.Normal('y', mu=??, sd=??, observed=data)
    trace_g = pm.sample(draws=1000) # backend = SQLite('test.sqlite') - Does not work

az.plot_trace(trace_g)
pm.model_to_graphviz(model_g)    

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC <br>
# MAGIC <hr style="border:2px green solid "> </hr>
# MAGIC 
# MAGIC ##### Note on scalablility 
# MAGIC If your trace information is too big as a result of too many variables or the model being large, you do not want to store this in memory since it can overrun the machine memory. Persisting this in a DB will allow you to reload it and inspect it at a later time as well. For each run, it appends the samples to the DB (or file if not deleted).

# COMMAND ----------

help(pm.backends)

# COMMAND ----------

from pymc3.backends.sqlite import load

with model_g:
    #trace = pm.backends.text.load('./mcmc')
    trace = pm.backends.sqlite.load('./mcmc.sqlite')
    
print(len(trace['??']))

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC <br>
# MAGIC <hr style="border:2px green solid "> </hr>
# MAGIC 
# MAGIC #### Pairplot for Correlations
# MAGIC 
# MAGIC Use a pairplot of the parameters to ensure that there are no correlations that would adversely affect the sampling process.

# COMMAND ----------

az.plot_pair(trace_g, kind='kde', fill_last=False)

# COMMAND ----------

az.summary(trace_g)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Posterior Predictive Check
# MAGIC 
# MAGIC We can draw samples from the inferred posterior distribution to check to see how they line up with the observed values. Below, we draw 100 samples of length corresponding to that of the data from this posterior. You are returned a dictionary for each of the observed variables in the model. 

# COMMAND ----------

y_pred_g = pm.sample_posterior_predictive(trace_g, 100, model_g)
print("Shape of the sampled variable y and data ",np.shape(y_pred_g['y']), len(data))

# COMMAND ----------

y_pred_g['y'][0]

# COMMAND ----------

# MAGIC %md
# MAGIC You can also plot the distribution of these samples by passing this variable 'y_pred_g' as shown below. Setting `mean=True` in the call to `plot_ppc` computes the mean distribution of the 100 sampled distributions and plots it as well.

# COMMAND ----------

data_ppc = az.from_pymc3(trace=trace_g, posterior_predictive=y_pred_g)
ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=True)
ax[0].legend(fontsize=15)

# COMMAND ----------

# MAGIC %md
# MAGIC Two things can be noted here:
# MAGIC 
# MAGIC * The mean distribution of the samples from the posterior predictive distribution is close to the distribution of the observed data but the mean of this mean distribution is slightly shifted to the right.
# MAGIC 
# MAGIC * Also, the variance of the samples; whether we can say qualitatively that this is acceptable or not depends on the problem. In general, the more representative data points available to us, the lower the variance.
# MAGIC 
# MAGIC Another thing to note here is that we modeled this problem using a Gaussian distribution, however we have some outliers that need to be accounted for which we cannot do well with a Gaussian distribution. We will see below how to use a Student's t-distribution for that.

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC <br>
# MAGIC <hr style="border:2px solid blue"> </hr>
# MAGIC 
# MAGIC ### Robust Models with a Student's t-Distribution
# MAGIC 
# MAGIC As mentioned in the previous section, one of the issues with assuming a Gaussian distribution is the assumption of finite variance. When you have observed data that lies outside this 'boundary', a Gaussian distribution is not a good fit and PyMC3, and other MCMC-based tools will be unable to reconcile these differences appropriately.
# MAGIC 
# MAGIC This distribution is parameterized by the following:
# MAGIC 
# MAGIC * ?? corresponds to the mean of the distribution
# MAGIC     
# MAGIC * ?? is the scale and corresponds roughly to the standard deviation
# MAGIC 
# MAGIC * ?? is the degrees of freedom and takes values between 0 and $\infty$. The degrees of freedom corresponds to the number of independent observations minus 1. When the sample size is 8, the t-distribution used to model this would have degrees of freedom set to 7. A value of 1 corresponds to the Cauchy distribution and indicates heavy tails, while infinity corresponds to a Normal distribution. 
# MAGIC 
# MAGIC The probability density function for a zero-centered Student's t-distribution with scale set to one is given by:
# MAGIC 
# MAGIC $p(t) = \dfrac{\gamma ((v+1) / 2)}{\sqrt{v \pi} \gamma (v/2)} (1 + \dfrac{t^2}{v})^{-(v+1)/2}$
# MAGIC 
# MAGIC In this case, the mean of the distribution is 0 and the variance is given by ??/(?? - 2).
# MAGIC 
# MAGIC Now let us model the same problem with this distribution instead of a Normal.

# COMMAND ----------

with pm.Model() as model_t:
    ?? = pm.Uniform('??', 40, 75) # mean
    ?? = pm.HalfNormal('??', sd=10)
    ?? = pm.Exponential('??', 1/30)
    y = pm.StudentT('y', mu=??, sd=??, nu=??, observed=data)
    trace_t = pm.sample(1000)

az.plot_trace(trace_t)
pm.model_to_graphviz(model_t)    

# COMMAND ----------

# MAGIC %md
# MAGIC Using a student's t-distribution we notice that the outliers are captured more accurately now and the model fits better.

# COMMAND ----------

# Using a student's t distribution we notice that the outliers are captured more 
# accurately now and the model fits better
y_ppc_t = pm.sample_posterior_predictive(
    trace_t, 100, model_t, random_seed=123)
y_pred_t = az.from_pymc3(trace=trace_t, posterior_predictive=y_ppc_t)
az.plot_ppc(y_pred_t, figsize=(12, 6), mean=True)
ax[0].legend(fontsize=15)
plt.xlim(40, 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading - Bayesian Estimation to Determine the Effectiveness of Drugs 
# MAGIC 
# MAGIC https://docs.pymc.io/notebooks/BEST.html

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC <br>
# MAGIC <hr style="border:2px solid blue"> </hr>
# MAGIC 
# MAGIC ### Hierarchical Models or Multilevel Models
# MAGIC 
# MAGIC Suppose we want to perform an analysis of water quality in a state and information is available from each district in the state. There are two ways to model this now: 
# MAGIC 
# MAGIC * We can study each district separately, however we lose information especially if there is insufficient data for some districts. But we get a more detailed model per district.
# MAGIC 
# MAGIC * The second option is to combine all the data and estimate the water quality of the state as a whole, i.e. a pooled model. We have more data but we lose granular information about each district.
# MAGIC 
# MAGIC The hierarchical model combines both of these options, by sharing information between the districts using hyperpriors that are priors over the parameter priors. In other words, instead of setting the prior parameters (or hyperparameters) to a constant value, we draw it from another prior distribution called the hyperprior. This hyperprior is shared among all the districts, and as a result information is shared between all the groups in the data.
# MAGIC 
# MAGIC #### Problem Statement
# MAGIC 
# MAGIC We measure the water samples for three districts, and we collect 30 samples for each district. The data is simply a binary value that indicates whether the water is contaminated or not. We count the number of samples that have contamination below the acceptable levels. We generate three arrays:
# MAGIC 
# MAGIC * N_samples - The total number of samples collected for each district or group
# MAGIC * G_samples - The number of good samples or samples with contamination levels below a certain threshold
# MAGIC * group_idx - The id for each district or group
# MAGIC 
# MAGIC #### Artifically generate the data

# COMMAND ----------

N_samples = [30, 30, 30] # Total number of samples collected
G_samples = [18, 18, 18] # Number of samples with water contamination 
                         # below accepted levels
# Create an ID for each of the 30 + 30 + 30 samples - 0,1,2 to indicate that they
# belong to different groups
group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]-G_samples[i]]))


# COMMAND ----------

# ID per sample
group_idx

# COMMAND ----------

data

# COMMAND ----------

# MAGIC %md
# MAGIC #### The Sampling Model
# MAGIC 
# MAGIC The scenario presented here is essentially a binary classification problem that can be modeled using a Bernoulli distribution. The parameter of the Bernoulli distribution is a vector corresponding to each group (\\(\theta_1, \theta_2, \theta_3\\)) and indicates the probability of getting a good sample (in each group). Since this is a hierarchical model, each group shares information and as a result the parameter of Group 1 can be influenced by the samples in Group 2 and 3. This is what makes hierarchical modeling so powerful. 
# MAGIC 
# MAGIC The process of generating our samples looks like the following. If we start from the last equation and work our way up, we can see that $\theta_i$ and $y_i$ are similar to a pooled model except that the beta prior takes parameters $\alpha$ and $\beta$ instead of constant values. These parameters now have hyperpriors applied to them using the parameters $\mu$ and *k* which are assumed to be distributed using a beta distribution and a half-Normal distribution respectively. Note that $\alpha$ and $\beta$ are indirectly computed from the terms \\(\mu\\) and *k* here. \\(\mu\\) affects the mean of the beta distribution and increasing *k* makes the beta distribution more concentrated. This parameterization is more efficient than the direct parameterization in terms of $\alpha_i$ and $\beta_i$.
# MAGIC 
# MAGIC 
# MAGIC $$ \mu \sim Beta(\alpha_p, \beta_p)  $$
# MAGIC $$ k \sim | Normal(0,\sigma_k) | $$
# MAGIC $$  \alpha =  \mu * k $$
# MAGIC $$  \beta = (1 - \mu) * k $$
# MAGIC $$  \theta_i \sim Beta(\alpha, \beta) $$
# MAGIC $$  y_i \sim Bern(\theta_i) $$

# COMMAND ----------

def get_hyperprior_model(data, N_samples, group_idx):
    with pm.Model() as model_h:
        ?? = pm.Beta('??', 1., 1.) # hyperprior
        ?? = pm.HalfNormal('??', 10) # hyperprior
        alpha = pm.Deterministic('alpha', ??*??)
        beta = pm.Deterministic('beta', (1.0-??)*??)
        ?? = pm.Beta('??', alpha=alpha, beta=beta, shape=len(N_samples)) # prior, len(N_samples) = 3
        y = pm.Bernoulli('y', p=??[group_idx], observed=data)
        trace_h = pm.sample(2000)
    az.plot_trace(trace_h)
    print(az.summary(trace_h))
    return(model_h)    
    
model = get_hyperprior_model(data, N_samples, group_idx)
pm.model_to_graphviz(model)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Shrinkage
# MAGIC 
# MAGIC Shrinkage refers to the phenomenon of sharing information among the groups through the use of hyperpriors. Hierarchical models can therefore be considered partially pooled models since information is shared among the groups so we move away from extreme values for the inferred parameters. This is good for two scenarios:
# MAGIC 
# MAGIC * If we have outliers (or poor quality data) in our data groups. 
# MAGIC * If we do not have a lot of data. 
# MAGIC 
# MAGIC In hierarchical models, the groups are neither independent (unpooled) nor do we clump all the data together (pooled) without accounting for the differences in the groups. 
# MAGIC 
# MAGIC We can look at three cases below as examples to illustrate the benefits of a hierarchical model. We keep the total number of samples and groups the same as before, however we vary the number of good samples in each group. When there are significant differences in the number of good samples within the groups, the behavior is different from what we see in an independent model. Averages win and extreme values are avoided.
# MAGIC 
# MAGIC The values of G_samples are changed to have the following values
# MAGIC 
# MAGIC 1. [5,5,5]
# MAGIC 2. [18,5,5]
# MAGIC 3. [18,18,1]
# MAGIC 
# MAGIC Note how the values of the three $\theta$s change as we change the values of G_samples.

# COMMAND ----------

# Case 1
N_samples = [30, 30, 30] # Total number of samples collected
G_samples = [5, 5, 5] # Number of samples with water contamination 
                         # below accepted levels
# Create an id for each of the 30 + 30 + 30 samples - 0,1,2 to indicate that they
# belong to different groups
group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]-G_samples[i]]))

model = get_hyperprior_model(data, N_samples, group_idx)
model

# COMMAND ----------

# Case 2 - The value of theta_1 is now smaller compared to our original case
N_samples = [30, 30, 30] # Total number of samples collected
G_samples = [18, 5, 5] # Number of samples with water contamination 
                         # below accepted levels
# Create an id for each of the 30 + 30 + 30 samples - 0,1,2 to indicate that they
# belong to different groups
group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]-G_samples[i]]))

get_hyperprior_model(data, N_samples, group_idx)

# COMMAND ----------

# Case 3 - Value of theta_3 is not as small as it would have been if it were estimated individually
N_samples = [30, 30, 30] # Total number of samples collected
G_samples = [18, 18, 1] # Number of samples with water contamination 
                         # below accepted levels
# Create an id for each of the 30 + 30 + 30 samples - 0,1,2 to indicate that they
# belong to different groups
group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]-G_samples[i]]))

get_hyperprior_model(data, N_samples, group_idx)

# COMMAND ----------

# MAGIC %md
# MAGIC ### GRADED EVALUATION (18 mins)
# MAGIC 
# MAGIC 1. According to the Central Limit Theorem, the mean of the sample means tends to the true population mean as the number of samples increase
# MAGIC 
# MAGIC     a. True 
# MAGIC 
# MAGIC     b. False
# MAGIC     
# MAGIC     
# MAGIC 2. Many real-world phenomena are averages of various factors, hence it is reasonable to use a Gaussian distribution to model them
# MAGIC 
# MAGIC     a. True
# MAGIC     
# MAGIC     b. False 
# MAGIC     
# MAGIC     
# MAGIC 3. What type of distribution is better suited to modeling positive values?
# MAGIC 
# MAGIC     a. Normal
# MAGIC     
# MAGIC     b. Half-normal
# MAGIC     
# MAGIC     
# MAGIC 4. Posterior predictive checks can be used to verify that the inferred distribution is similar to the observed data
# MAGIC 
# MAGIC     a. True
# MAGIC     
# MAGIC     b. False
# MAGIC     
# MAGIC     
# MAGIC 5. Which distribution is better suited to model data that has a lot of outliers?
# MAGIC 
# MAGIC     a. Gaussian distribution
# MAGIC     
# MAGIC     b. Student's t-distribution
# MAGIC     
# MAGIC     
# MAGIC 6. Hierarchical models are beneficial in modeling data from groups where there might be limited data in certain groups
# MAGIC 
# MAGIC     a. True
# MAGIC     
# MAGIC     b. False
# MAGIC     
# MAGIC   
# MAGIC 7. Hierarchical models share information through hyperpriors
# MAGIC 
# MAGIC     a. True
# MAGIC     
# MAGIC     b. False

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC <br>
# MAGIC <hr style="border:2px solid blue"> </hr>
# MAGIC 
# MAGIC 
# MAGIC ### Linear Regression Again!
# MAGIC 
# MAGIC Let us generate some data for linear regression and plot it along with its density. 

# COMMAND ----------

np.random.seed(1)
N = 100

# Parameters
alpha_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)

# Input data drawn from a Normal distribution
x = np.random.normal(10, 1, N)

# Output generated from the input and the parameters
y_real = alpha_real + beta_real * x

# Add random noise to y
y = y_real + eps_real

# Plot the data
_, ax = plt.subplots(1,2, figsize=(8, 4))
ax[0].plot(x, y, 'C0.')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y', rotation=0)
ax[0].plot(x, y_real, 'k')
az.plot_kde(y, ax=ax[1])
ax[1].set_xlabel('y')
ax[1].set_ylabel('p(y)')
plt.tight_layout()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Inference of Parameters in Linear Regression

# COMMAND ----------

import pymc3 as pm

with pm.Model() as model_g:
    ?? = pm.Normal('??', mu=0, sd=10)
    ?? = pm.Normal('??', mu=0, sd=1)
    ?? = pm.HalfCauchy('??', 5) # Try changing this to a half normal, half cauchy has fatter tails
    ?? = pm.Deterministic('??', ?? + ?? * x)
    y_pred = pm.Normal('y_pred', mu=??, sd=??, observed=y)
    trace_g = pm.sample(2000, tune=1000)
    
az.plot_trace(trace_g, var_names=['??', '??', '??']) # if you have a lot of variables, explicitly specify
plt.figure()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Parameter Correlations

# COMMAND ----------

# Pairplot
az.plot_pair(trace_g, var_names=['??', '??'], plot_kwargs={'alpha': 0.1}) # Notice the diagonal shape

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualize the Uncertainty

# COMMAND ----------

plt.figure()
# Plot the true values
plt.plot(x, y, 'C0.') 

# Get the mean inferred values
alpha_m = trace_g['??'].mean() 
beta_m = trace_g['??'].mean()  

# Plot all draws to show the variance of the regression lines
draws = range(0, len(trace_g['??']), 10)
plt.plot(x, trace_g['??'][draws] + trace_g['??'][draws]* x[:, np.newaxis], c='lightblue', alpha=0.5)

# Plot the mean regression line
plt.plot(x, alpha_m + beta_m * x, c='teal', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')

plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Posterior Sampling

# COMMAND ----------

ppc = pm.sample_posterior_predictive(trace_g,
                                     samples=2000,
                                     model=model_g)


# Plot the posterior predicted samples, i.e. these are samples of predicted y for each original x in our data
az.plot_hpd(x, ppc['y_pred'], credible_interval=0.5, color='lightblue')

# Plot the true y values
plt.plot(x, y, 'b.') 

# Plot the mean regression line - from cell above
plt.plot(x, alpha_m + beta_m * x, c='teal') 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Mean-center the Data
# MAGIC 
# MAGIC Looking at the pairplot of $\alpha$ and $\beta$, one can notice the high degree of correlation between these two variables as indicated by the narrow joint density. This results in a parameter posterior space that is diagonally shaped, which is problematic for many samplers such as the Metropolis-Hastings MCMC sampler. One recommended approach to minimize this correlation is to center the independent variables. If \\(\bar{x}\\) is the mean of the data x then
# MAGIC 
# MAGIC $$\tilde{x} = x - \bar{x}$$
# MAGIC 
# MAGIC The advantage of this is twofold:
# MAGIC 
# MAGIC 1. The pivot point is the intercept when the slope changes
# MAGIC 2. The parameter posterior space is more circular
# MAGIC 
# MAGIC ##### Transformation
# MAGIC 
# MAGIC In order to center the data, the original equation for linear regression given by
# MAGIC 
# MAGIC $$y = \alpha + \beta x$$
# MAGIC 
# MAGIC has to be equivalent to the equation for the centered data
# MAGIC 
# MAGIC $$y = \tilde{\alpha} + \tilde{\beta}(x - \bar{x}) = \tilde{\alpha} - \tilde{\beta} \bar{x} + \tilde{\beta} x$$
# MAGIC 
# MAGIC 
# MAGIC ##### Recovering the data
# MAGIC 
# MAGIC This implies that we can recover the original intercept \\(\alpha\\) as
# MAGIC 
# MAGIC $$ \alpha = \tilde{\alpha} - \tilde{\beta} \bar{x}$$
# MAGIC 
# MAGIC and \\(\beta\\) as
# MAGIC 
# MAGIC $$ \beta = \tilde{\beta} $$
# MAGIC 
# MAGIC 
# MAGIC #### Standardize the data
# MAGIC 
# MAGIC You can also standardize the data by mean centering and dividing by the standard deviation
# MAGIC 
# MAGIC $$\tilde{x} = (x - \bar{x}) / \sigma_x$$

# COMMAND ----------

# MAGIC %md
# MAGIC #### Mean Centered - Broader Sampling Space

# COMMAND ----------

# Center the data
x_centered = x - x.mean()

with pm.Model() as model_g:
    ?? = pm.Normal('??', mu=0, sd=10)
    ?? = pm.Normal('??', mu=0, sd=1)
    ?? = pm.HalfCauchy('??', 5)
    ?? = pm.Deterministic('??', ?? + ?? * x_centered)
    y_pred = pm.Normal('y_pred', mu=??, sd=??, observed=y)
    ??_recovered = pm.Deterministic('??_recovered', ?? - ?? * x.mean())
    trace_g = pm.sample(2000, tune=1000)
    
az.plot_trace(trace_g, var_names=['??', '??', '??'])
plt.figure()
az.plot_pair(trace_g, var_names=['??', '??'], plot_kwargs={'alpha': 0.1})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Robust Linear Regression
# MAGIC 
# MAGIC We fitted our model parameters by assuming the data likelihood was a Normal distribution, however as we saw earlier this assumption suffers from not doing well with outliers. Our solution to this problem is the same, use a Student's t-distribution for the likelihood.
# MAGIC 
# MAGIC Here we look at the [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet), which is a set of 4 data sets. They have similar statistical properties even though they look very different and were used to illustrate the need to visualize the data along with the effect of outliers. Our intended goal is the same, to model data with outliers and assess the sensitivity of the model to these outliers.

# COMMAND ----------

import seaborn as sns
from scipy import stats

# Load the example dataset for Anscombe's quartet
df = sns.load_dataset("anscombe")
df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Plot the 4 Subgroups in the Data

# COMMAND ----------

x_0 = df[df.dataset == 'I']['x'].values
y_0 = df[df.dataset == 'I']['y'].values
x_1 = df[df.dataset == 'II']['x'].values
y_1 = df[df.dataset == 'II']['y'].values
x_2 = df[df.dataset == 'III']['x'].values
y_2 = df[df.dataset == 'III']['y'].values
x_3 = df[df.dataset == 'IV']['x'].values
y_3 = df[df.dataset == 'IV']['y'].values
_, ax = plt.subplots(2, 2, figsize=(12,8), sharex=True, sharey=True)


print("Mean of x values in all groups -- ",x_0.mean(), x_1.mean(), x_2.mean(), x_3.mean())
print("Mean of y values in all groups -- ",y_0.mean(), y_1.mean(), y_2.mean(), y_3.mean())
print("Mean of x values in all groups -- ",x_0.var(), x_1.var(), x_2.var(), x_3.var())
print("Mean of y values in all groups -- ",y_0.var(), y_1.var(), y_2.var(), y_3.var())

ax = np.ravel(ax)
ax[0].scatter(x_0, y_0)
sns.regplot(x_0, y_0, ax=ax[0])
ax[0].set_title('Group I')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y', rotation=0, labelpad=15)

ax[1].scatter(x_1, y_1)
sns.regplot(x_1, y_1, ax=ax[1])
ax[1].set_title('Group II')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y', rotation=0, labelpad=15)

ax[2].scatter(x_2, y_2)
sns.regplot(x_2, y_2, ax=ax[2])
ax[2].set_title('Group III')
ax[2].set_xlabel('x')
ax[2].set_ylabel('y', rotation=0, labelpad=15)

ax[3].scatter(x_3, y_3)
sns.regplot(x_3, y_3, ax=ax[3])
ax[3].set_title('Group IV')
ax[3].set_xlabel('x')
ax[3].set_ylabel('y', rotation=0, labelpad=15)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Plot Data Group 3 and its Kernel Density

# COMMAND ----------

x_2 = x_2 - x_2.mean()
_, ax = plt.subplots(1, 2, figsize=(10, 5))
beta_c, alpha_c = stats.linregress(x_2, y_2)[:2]
ax[0].plot(x_2, (alpha_c + beta_c * x_2), 'k',
           label=f'y ={alpha_c:.2f} + {beta_c:.2f} * x')
ax[0].plot(x_2, y_2, 'C0o')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y', rotation=0)
ax[0].legend(loc=0)
az.plot_kde(y_2, ax=ax[1], rug=True)
ax[1].set_xlabel('y')
ax[1].set_yticks([])
plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model using a Student's t Distribution

# COMMAND ----------

with pm.Model() as model_t:
    ?? = pm.Normal('??', mu=y_2.mean(), sd=1)
    ?? = pm.Normal('??', mu=0, sd=1)
    ?? = pm.HalfNormal('??', 5)
    ??_ = pm.Exponential('??_', 1/29)
    ?? = pm.Deterministic('??', ??_ + 1) # shifting the exponential to avoid values close to 0
    y_pred = pm.StudentT('y_pred', mu=?? + ?? * x_2,
                         sd=??, nu=??, observed=y_2)
    trace_t = pm.sample(2000)
    
alpha_m = trace_t['??'].mean()
beta_m = trace_t['??'].mean()
plt.plot(x_2, alpha_m + beta_m * x_2, c='k', label='robust')
plt.plot(x_2, y_2, '*')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend(loc=2)
plt.tight_layout()

pm.model_to_graphviz(model_t)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hierarchical Linear Regression
# MAGIC 
# MAGIC We want to use the same hierarchical or multilevel modeling technique that we discussed earlier, for linear regression problems as well. As mentioned above, this is particularly useful when presented with imbalanced subgroups of sparse data. In this example, we create data with 8 subgroups. In this data, 7 of the subgroups have 20 data points and the last one has a single data point. 
# MAGIC 
# MAGIC The data for all the 8 groups are generated from a normal distribution of mean 10 and a standard deviation of 1. The parameters for the linear model are generated from the normal and beta distributions.
# MAGIC 
# MAGIC #### Data Generation

# COMMAND ----------

N = 20
M = 8
idx = np.repeat(range(M-1), N)
idx = np.append(idx, 7)

np.random.seed(314)
alpha_real = np.random.normal(4, 1, size=M) 
beta_real = np.random.beta(7, 1, size=M)
eps_real = np.random.normal(0, 0.5, size=len(idx))
print("Alpha parameters ", alpha_real )

y_m = np.zeros(len(idx))
x_m = np.random.normal(10, 1, len(idx))
y_m = alpha_real[idx] + beta_real[idx] * x_m + eps_real
_, ax = plt.subplots(2, 4, figsize=(12,8), sharex=True, sharey=True)

ax = np.ravel(ax)
j, k = 0, N
for i in range(M):
    ax[i].scatter(x_m[j:k], y_m[j:k])
    ax[i].set_xlabel(f'x_{i}')
    ax[i].set_ylabel(f'y_{i}', rotation=0, labelpad=15)
    ax[i].set_xlim(6, 15)
    ax[i].set_ylim(7, 17)     
    j += N
    k += N
plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Non-hierarchical Model
# MAGIC 
# MAGIC We build a non-hierarchical model first for comparison. We also mean-center the data for ease of convergence. Note how the obtained $\alpha$ and $\beta$ values vary for each group, particularly the scale of the last one, which is really off.

# COMMAND ----------

# Center the data
x_centered = x_m - x_m.mean()

with pm.Model() as unpooled_model:
    # Note the M prior parameters for the M groups
    ??_tmp = pm.Normal('??_tmp', mu=2, sd=5, shape=M)
    ?? = pm.Normal('??', mu=0, sd=10, shape=M)
    ?? = pm.HalfCauchy('??', 5)
    ?? = pm.Exponential('??', 1/30)
    
    y_pred = pm.StudentT('y_pred', mu=??_tmp[idx] + ??[idx] * x_centered, sd=??, nu=??, observed=y_m)
    # Rescale alpha back - after x had been centered the computed alpha is different from the original alpha
    ?? = pm.Deterministic('??', ??_tmp - ?? * x_m.mean())
    trace_up = pm.sample(2000)

az.plot_trace(trace_up)
plt.figure()
az.plot_forest(trace_up, var_names=['??', '??'], combined=True)
az.summary(trace_up)

pm.model_to_graphviz(unpooled_model)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Hierarchical Model
# MAGIC 
# MAGIC We set hyperpriors on the \\(\alpha\\) and \\(\beta\\) parameters. To be more precise, the hyperpriors are applied to the scaled version of \\(\alpha\\), i.e. \\(\alpha_{tmp}\\).

# COMMAND ----------

with pm.Model() as hierarchical_model:
    # Hyperpriors - we add these instead of setting the prior values to a constant
    # Note that there exists only one hyperprior  for all M groups, shared hyperprior
    ??_??_tmp = pm.Normal('??_??_tmp', mu=100, sd=1) # try changing these hyperparameters
    ??_??_tmp = pm.HalfNormal('??_??_tmp', 10) # try changing these hyperparameters
    ??_?? = pm.Normal('??_??', mu=10, sd=2) # reasonable changes do not have an impact
    ??_?? = pm.HalfNormal('??_??', sd=5)
    
    # priors - note that the prior parameters are no longer a constant
    ??_tmp = pm.Normal('??_tmp', mu=??_??_tmp, sd=??_??_tmp, shape=M)
    ?? = pm.Normal('??', mu=??_??, sd=??_??, shape=M)
    ?? = pm.HalfCauchy('??', 5)
    ?? = pm.Exponential('??', 1/30)
    y_pred = pm.StudentT('y_pred',
                         mu=??_tmp[idx] + ??[idx] * x_centered,
                         sd=??, nu=??, observed=y_m)
    ?? = pm.Deterministic('??', ??_tmp - ?? * x_m.mean())
    ??_?? = pm.Deterministic('??_??', ??_??_tmp - ??_?? *
                           x_m.mean())
    ??_?? = pm.Deterministic('??_sd', ??_??_tmp - ??_?? * x_m.mean())
    trace_hm = pm.sample(1000)

az.plot_forest(trace_hm, var_names=['??', '??'], combined=True)

_, ax = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True,
                     constrained_layout=True)
ax = np.ravel(ax)
j, k = 0, N
x_range = np.linspace(x_m.min(), x_m.max(), 10)
for i in range(M):
    ax[i].scatter(x_m[j:k], y_m[j:k])
    ax[i].set_xlabel(f'x_{i}')
    ax[i].set_ylabel(f'y_{i}', labelpad=17, rotation=0)
    alpha_m = trace_hm['??'][:, i].mean()
    beta_m = trace_hm['??'][:, i].mean()
    ax[i].plot(x_range, alpha_m + beta_m * x_range, c='k',
               label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
    plt.xlim(x_m.min()-1, x_m.max()+1)
    plt.ylim(y_m.min()-1, y_m.max()+1)
    j += N
    k += N

pm.model_to_graphviz(hierarchical_model)

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC <br>
# MAGIC <hr style="border:2px solid blue"> </hr>
# MAGIC 
# MAGIC ### Polynomial Regression for Nonlinear Data
# MAGIC 
# MAGIC What happens when the data is inherently nonlinear? It is more appropriate to use non-linear combinations of the inputs. This could be in the form of higher order terms such as \\(x^2, x^3\\) or it could use basis functions such as the cosine function, \\(cos(x)\\).
# MAGIC 
# MAGIC #### Data Generation
# MAGIC 
# MAGIC Use the values from the dataset in Anscombe's quartet we used earlier as our non-linear data. We will use the regression model given by
# MAGIC 
# MAGIC $$ y = \alpha + \beta_1 * x_{centered} + \beta_2 * x_{centered}^2 $$

# COMMAND ----------

x_1_centered = x_1 - x_1.mean()
plt.scatter(x_1_centered, y_1)
plt.xlabel('x')
plt.ylabel('y', rotation=0)

plt.figure()
x_0_centered = x_0 - x_0.mean()
plt.scatter(x_0_centered, y_0)
plt.xlabel('x')
plt.ylabel('y', rotation=0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inference on Data I

# COMMAND ----------

with pm.Model() as model_poly:
    ?? = pm.Normal('??', mu=y_1.mean(), sd=1)
    ??1 = pm.Normal('??1', mu=0, sd=1)
    ??2 = pm.Normal('??2', mu=0, sd=1)
    ?? = pm.HalfCauchy('??', 5)
    mu = ?? + ??1 * x_1_centered + ??2 * x_1_centered**2
    y_pred = pm.Normal('y_pred', mu=mu, sd=??, observed=y_1)
    trace = pm.sample(2000, tune=2000)
    
x_p = np.linspace(-6, 6)
y_p = trace['??'].mean() + trace['??1'].mean() * x_p + trace['??2'].mean() * x_p**2
plt.scatter(x_1_centered, y_1)
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.plot(x_p, y_p, c='C1')



# COMMAND ----------

# MAGIC %md
# MAGIC #### Inference on Data II

# COMMAND ----------

with pm.Model() as model_poly:
    ?? = pm.Normal('??', mu=y_0.mean(), sd=1)
    ??1 = pm.Normal('??1', mu=0, sd=1)
    ??2 = pm.Normal('??2', mu=0, sd=1)
    ?? = pm.HalfCauchy('??', 5)
    mu = ?? + ??1 * x_0_centered + ??2 * x_0_centered**2
    y_pred = pm.Normal('y_pred', mu=mu, sd=??, observed=y_0)
    trace = pm.sample(2000)
    
x_p = np.linspace(-6, 6)
y_p = trace['??'].mean() + trace['??1'].mean() * x_p + trace['??2'].mean() * x_p**2
plt.scatter(x_0_centered, y_0)
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.plot(x_p, y_p, c='C1')

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC <br>
# MAGIC <hr style="border:2px solid blue"> </hr>
# MAGIC 
# MAGIC ### Multiple Linear Regression
# MAGIC 
# MAGIC In Multiple Linear Regression, there is more than one independent variable to predict the outcome of one dependent variable.
# MAGIC 
# MAGIC $$ y = \alpha + \overrightarrow{\beta} \cdot \overrightarrow{X} $$
# MAGIC 
# MAGIC #### Data Generation
# MAGIC 
# MAGIC The example below generates two-dimensional data for X. It plots the variation of 'y' with each component of X in the top two figures. The bottom figure indicates the correlation of the two components of X.

# COMMAND ----------

np.random.seed(314)

# N is the total number of observations 
N = 100
# m is 2, the number of independent variables
alpha_real = 2.5
beta_real = [0.9, 1.5]
eps_real = np.random.normal(0, 0.5, size=N)

# X is # n x m
X = np.array([np.random.normal(i, j, N) for i, j in zip([10, 2], [1, 1.5])]).T 
X_mean = X.mean(axis=0, keepdims=True)
X_centered = X - X_mean
y = alpha_real + np.dot(X, beta_real) + eps_real

def scatter_plot(x, y):
    plt.figure(figsize=(10, 10))
    for idx, x_i in enumerate(x.T):
        plt.subplot(2, 2, idx+1)
        plt.scatter(x_i, y)
        plt.xlabel(f'x_{idx+1}')
        plt.ylabel(f'y', rotation=0)
    plt.subplot(2, 2, idx+2)
    plt.scatter(x[:, 0], x[:, 1])
    plt.xlabel(f'x_{idx}')
    plt.ylabel(f'x_{idx+1}', rotation=0)

scatter_plot(X_centered, y)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inference
# MAGIC 
# MAGIC This code is very similar to what we have already seen, the only real difference being the dimensionality of the coefficients and the inputs. Something you would notice is that as the number of unknowns increase, the uncertainty associated with our inferences become larger. It is beneficial to have more accurate priors in this situation.

# COMMAND ----------

with pm.Model() as model_mlr:
    ??_tmp = pm.Normal('??_tmp', mu=2, sd=2) # Try changing the prior distribution
    ?? = pm.Normal('??', mu=0, sd=5, shape=2) # Note the shape of beta
    ?? = pm.HalfCauchy('??', 5)
    ?? = ??_tmp + pm.math.dot(X_centered, ??)
    ?? = pm.Deterministic('??', ??_tmp - pm.math.dot(X_mean, ??))
    y_pred = pm.Normal('y_pred', mu=??, sd=??, observed=y)
    trace = pm.sample(2000, tune=1000)
    
az.summary(trace)

# COMMAND ----------

pm.model_to_graphviz(model_mlr)

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC <br>
# MAGIC <hr style="border:2px solid blue"> </hr>
# MAGIC 
# MAGIC ### Logistic Regression

# COMMAND ----------

# MAGIC %md
# MAGIC While everything we have seen so far involved regression, the same ideas can be applied to a classification task as well. We use the logistic regression model to perform this classification here. The name 'regression' is due to the fact that the model outputs class probabilities as numbers which is then converted into classes using a decision boundary. There are many ways to select an appropriate decision boundary, a few of which were covered in Course 1 and Course 2.
# MAGIC 
# MAGIC #### Inverse Link function
# MAGIC 
# MAGIC At this point it is a good idea to bring up the concept of a inverse link function, which takes the form
# MAGIC 
# MAGIC $\theta = f(\alpha + \beta x)$
# MAGIC 
# MAGIC Here 'f' is called the inverse link function, the term inverse refers to the fact that the function is applied to the right hand side of the equation. In a linear regression, this inverse link function is the identity function. In the case of a linear regression model, the value 'y' at any point 'x' is modeled as the mean of a Gaussian distribution centered at the point (x,y). The error as a result of the true 'y' and the estimated 'y' are modeled with the standard deviation of this Gaussian at that point (x,y). Now think about the scenario where this is not appropriately modeled using a Gaussian. A classification problem is a perfect example of such a scenario where the discrete classes are not modeled well as a Gaussian and hence we can't use this distribution to model the mean of those classes. As a result, we would like to convert the output of $\alpha + \beta x$ to some other range of values that are more appropriate to the problem being modeled, which is what the link function intends to do.
# MAGIC 
# MAGIC #### Logistic function
# MAGIC 
# MAGIC The logistic function is defined as the function
# MAGIC 
# MAGIC $logistic(x) = \dfrac{1}{1 + \exp{(-x)}}$
# MAGIC 
# MAGIC This is also called the sigmoid function and it restricts the value of the output to the range [0,1].

# COMMAND ----------

x = np.linspace(-5,5)
plt.plot(x, 1 / (1 + np.exp(-x)))
plt.xlabel('x')
plt.ylabel('logistic(x)')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Example using the Iris data
# MAGIC 
# MAGIC The simplest example using a logistic regression model is one that can be used to identify two classes. If you are given a set of independent variables that are features which correspond to an output dependent variable that is a class, you can build a model to learn the relationship between the features and the output classes. This is done with the help of the logistic function which acts as the inverse link function to relate the features to the output class. 
# MAGIC 
# MAGIC $\theta = logistic(\alpha + \beta x)$
# MAGIC 
# MAGIC If it is a two-class problem (binary classification), the output variable can be represented by a Bernoulli distribution.
# MAGIC 
# MAGIC $y \sim Bern(\theta)$
# MAGIC 
# MAGIC The mean parameter $\theta$ is now given by the regression equation $logistic(\alpha + \beta x)$. In regular linear regression, this parameter was drawn from a Gaussian distribution. In the case of the coin-flip example the data likelihood was represented by a Bernoulli distribution, (the parameter $\theta$ was drawn from a Beta prior distribution there), similarly we have output classes associated with every observation here.
# MAGIC 
# MAGIC We load the iris data from scikit learn and
# MAGIC 
# MAGIC * Plot the distribution of the three classes for two of the features. 
# MAGIC 
# MAGIC * We also perform a pairplot to visualize the correlation of each feature with every other feature. The diagonal of this plot shows the distribution of the three classes for that feature.
# MAGIC 
# MAGIC * Correlation plot of just the features. This can be visually cleaner and cognitively simpler to comprehend.

# COMMAND ----------

import pymc3 as pm
import sklearn
import numpy as np
import graphviz
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
from sklearn import datasets
df = datasets.load_iris()
iris_data = pd.DataFrame(df['data'], columns=df['feature_names'])
iris_data['target'] = df['target']
seaborn.stripplot(x='target', y='sepal length (cm)', data=iris_data, jitter=False)
plt.figure()
seaborn.stripplot(x='target', y='petal length (cm)', data=iris_data, jitter=False)
plt.figure()
seaborn.pairplot(iris_data, hue='target', diag_kind='kde')
plt.figure()
corr = iris_data.query("target == (0,1)").loc[:, iris_data.columns != 'target'].corr() 
mask = np.tri(*corr.shape).T 
seaborn.heatmap(corr.abs(), mask=mask, annot=True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC You would notice that some of the variables have a high degree of correlation from the correlation plot. One approach is to eliminate one of the correlated variables. The second option is to mean-center and use a weakly-informative prior such as a Students t-distribution for all variables that are not binary. The scale parameter can be adjusted for the range of expected values for these variables and the normality parameter is recommended to be between 3 and 7. (Source: Andrew Gelman and the Stan team)

# COMMAND ----------

df['target_names']

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inference
# MAGIC 
# MAGIC We use a **single feature**, the sepal length, to learn a decision boundary between the **first two classes** in the iris data (0,1).
# MAGIC 
# MAGIC In this case, the decision boundary is defined to be the value of 'x' when 'y' = 0.5. We won't go over the derivation here, but this turns out to be \\(-\alpha / \beta\\). However, this value was chosen under the assumption that the midpoint of the class values are a good candidate for separating the classes, but this does not have to be the case.

# COMMAND ----------

# Select the first two classes for a binary classification problem
df = iris_data.query("target == (0,1)")
y_0 = df.target
x_n = 'sepal length (cm)' 
x_0 = df[x_n].values
x_c = x_0 - x_0.mean()

# COMMAND ----------

import pymc3 as pm
import arviz as az

with pm.Model() as model_0:
    ?? = pm.Normal('??', mu=0, sd=10)
    ?? = pm.Normal('??', mu=0, sd=10)
    ?? = ?? + pm.math.dot(x_c, ??)    
    ?? = pm.Deterministic('??', pm.math.sigmoid(??))
    bd = pm.Deterministic('bd', -??/??)
    yl = pm.Bernoulli('yl', p=??, observed=y_0)
    trace_0 = pm.sample(1000)

pm.model_to_graphviz(model_0)

# COMMAND ----------

az.summary(trace_0, var_names=["??","??","bd"])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualizing the Decision Boundary
# MAGIC 
# MAGIC * The classifier outputs, i.e. the y values are jittered to make it easier to visualize. 
# MAGIC * The solid green lines are the mean of the fitted \\(\theta\\) as a result of the sampling and inference process
# MAGIC * The transparent green lines indicate the 94% HPD (default values) for the fitted \\(\theta\\).
# MAGIC * The solid blue line indicates the decision boundary that is derived from the inferred values of the parameters using the equation \\(\alpha/\beta\\)
# MAGIC * The transparent blue indicates the HPD (94%) for the decision boundary.

# COMMAND ----------

theta = trace_0['??'].mean(axis=0)
idx = np.argsort(x_c)

# Plot the fitted theta
plt.plot(x_c[idx], theta[idx], color='teal', lw=3)
# Plot the HPD for the fitted theta
az.plot_hpd(x_c, trace_0['??'], color='teal')
plt.xlabel(x_n)
plt.ylabel('??', rotation=0)

# Plot the decision boundary
plt.vlines(trace_0['bd'].mean(), 0, 1, color='steelblue')
# Plot the HPD for the decision boundary
bd_hpd = az.hpd(trace_0['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='steelblue', alpha=0.5)
plt.scatter(x_c, np.random.normal(y_0, 0.02),
            marker='.', color=[f'C{x}' for x in y_0])

# use original scale for xticks
locs, _ = plt.xticks()
plt.xticks(locs, np.round(locs + x_0.mean(), 1))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Multiple Logistic Regression
# MAGIC 
# MAGIC The above example with a single feature can be extended to take **multiple features** or independent variables to separate the same **two classes**.

# COMMAND ----------

# Select the first two classes for a binary classification problem
df = iris_data.query("target == (0,1)")
y_0 = df.target
x_n = ['sepal length (cm)', 'sepal width (cm)']
# Center the data by subtracting the mean from both columns
df_c = df - df.mean() 
x_c = df_c[x_n].values

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC As we saw before, the equation for multiple logistic regression relating the $\theta$ parameter to the features can be written as 
# MAGIC 
# MAGIC $$\theta = logistic(\alpha + \beta_1 x_1 + \beta_2 x_2)$$
# MAGIC 
# MAGIC $$y \sim Bern(\theta)$$
# MAGIC 
# MAGIC This gives us a decision boundary, assuming y = 0.5 is a reasonable boundary, of 
# MAGIC 
# MAGIC $$x_2 = -\dfrac{\alpha}{\beta_2} - \dfrac{\beta_1}{\beta_2} x_1$$
# MAGIC 
# MAGIC Unlike the previous equation, this one represents a line for the variables \\(x_1\\) and \\(x_2\\) which separates the two-dimensional space occupied by \\(x_1\\) and \\(x_2\\). For higher dimensions, this decision boundary will be a hyperplane of dimension 'n-1' for a feature space of dimension 'n'.
# MAGIC 
# MAGIC #### Inference

# COMMAND ----------

with pm.Model() as model_1: 
    ?? = pm.Normal('??', mu=0, sd=10) 
    ?? = pm.Normal('??', mu=0, sd=2, shape=len(x_n)) 
    ?? = ?? + pm.math.dot(x_c, ??) 
    ?? = pm.Deterministic('??', 1 / (1 + pm.math.exp(-??))) 
    bd = pm.Deterministic('bd', -??/??[1] - ??[0]/??[1] * x_c[:,0])
    yl = pm.Bernoulli('yl', p=??, observed=y_0) 
    trace_0 = pm.sample(2000)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualization
# MAGIC 
# MAGIC We plot the HPD on the centered data, we have not scaled it back to the original range here.

# COMMAND ----------

idx = np.argsort(x_c[:,0]) 
bd = trace_0['bd'].mean(0)[idx] 
plt.scatter(x_c[:,0], x_c[:,1], c=[f'C{x}' for x in y_0]) 
plt.plot(x_c[:,0][idx], bd, color='steelblue'); 
az.plot_hpd(x_c[:,0], trace_0['bd'], color='steelblue')
plt.xlabel(x_n[0]) 
plt.ylabel(x_n[1])

# COMMAND ----------

pm.model_to_graphviz(model_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multiclass Classification
# MAGIC 
# MAGIC If we have more than two classes this becomes a multiclass problem. In this case we use the softmax function instead of the sigmoid function. The sigmoid function is a special case of the softmax for a two-class classification problem. The softmax function can be written as
# MAGIC 
# MAGIC $$ softmax(x_i) = \dfrac{\exp(x_i)}{\sum_k \exp(x_k)} $$
# MAGIC 
# MAGIC Earlier, we also used a Bernoulli distribution as the likelihood for our $\theta$ parameter, however now we sample from a categorical distribution.
# MAGIC 
# MAGIC $$\theta = softmax(\alpha + \beta x)$$
# MAGIC 
# MAGIC $$y \sim Categorical(\theta)$$
# MAGIC 
# MAGIC #### Logistic Regression for a Multiclass Problem 
# MAGIC 
# MAGIC We reuse the previous example, however we are going to use all the three classes in the data here along with all the features for maximum separability. The data is also standardized instead of just mean-centered here.

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import warnings
# MAGIC warnings.filterwarnings("ignore")
# MAGIC import arviz as az
# MAGIC import pymc3 as pm
# MAGIC import numpy as np
# MAGIC import graphviz
# MAGIC import pandas as pd
# MAGIC from matplotlib import pyplot as plt
# MAGIC import seaborn as sns
# MAGIC from sklearn import datasets
# MAGIC df = datasets.load_iris()
# MAGIC iris_data = pd.DataFrame(df['data'], columns=df['feature_names'])
# MAGIC iris_data['target'] = df['target']
# MAGIC y_s = iris_data.target
# MAGIC x_n = iris_data.columns[:-1]
# MAGIC x_s = iris_data[x_n]
# MAGIC x_s = (x_s - x_s.mean()) / x_s.std()
# MAGIC x_s = x_s.values

# COMMAND ----------

import theano as tt
tt.config.gcc.cxxflags = "-Wno-c++11-narrowing"

with pm.Model() as model_mclass:
    alpha = pm.Normal('alpha', mu=0, sd=5, shape=3)
    beta = pm.Normal('beta', mu=0, sd=5, shape=(4,3))
    ?? = pm.Deterministic('??', alpha + pm.math.dot(x_s, beta))
    ?? = tt.tensor.nnet.softmax(??)
    #?? = pm.math.exp(??)/pm.math.sum(pm.math.exp(??), axis=0)
    yl = pm.Categorical('yl', p=??, observed=y_s)
    trace_s = pm.sample(2000)

data_pred = trace_s['??'].mean(0)
y_pred = [np.exp(point)/np.sum(np.exp(point), axis=0) for point in data_pred]
az.plot_trace(trace_s, var_names=['alpha'])
f'{np.sum(y_s == np.argmax(y_pred, axis=1)) / len(y_s):.2f}'


# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC <br>
# MAGIC <hr style="border:2px solid blue"> </hr>
# MAGIC 
# MAGIC ### Inferring Rate Change with a Poisson Distribution
# MAGIC 
# MAGIC Discrete variables that represents count data can be handled using a Poisson distribution. The key element here is that it is the number of events happening in a given interval of time. The events are supposed to be independent and   the distribution is parameterized using a single value called the rate parameter. This corresponds to and controls both the mean and the variance of the distribution. One implication of this is that the higher the mean, the larger the variance of the distribution which can be a limitation for some phenomena. A higher value of the rate parameter indicates a higher likelihood of getting larger values from our distribution. It is represented by
# MAGIC 
# MAGIC $f(x) = e^{-\mu} \mu^x / x!$
# MAGIC 
# MAGIC * The mean rate is represented by $\mu$
# MAGIC * x is a positive integer that represents the number of events that can happen
# MAGIC 
# MAGIC If you recall from the discussion of the binomial distribution, that can also be used to model the probability of the number of successes out of 'n' trials. The Poisson distribution is a special case of this binomial distribution and is used when the trials far exceed the number of successes. 
# MAGIC 
# MAGIC #### Poisson Distribution Example
# MAGIC 
# MAGIC In the following example we look at a time-varying rate phenomena, consider the observations as the number of COVID-19 cases per day. We observe cases for 140 days, however due to some interventional measures put in place it is suspected that the number of cases per day have gone down. If we assume that the number of cases can be modeled using a Poisson distribution, then this implies that there are two rates $\lambda_1$ and $\lambda_2$, and we can try to find where this rate-switch happens (time $\tau$). 
# MAGIC 
# MAGIC We don't really know a lot about these rates, so we select a prior for both which can be from an Exponential, Gamma or Uniform distributions. Both the Exponential and Gamma distributions work better than the Uniform distribution since the Uniform distribution is the least informative. As usual, with enough observations one can even get away with a Uniform prior. Since we have no information regarding $\tau$, we select a Uniform prior distribution for that.
# MAGIC 
# MAGIC In the example below, try varying the following
# MAGIC 
# MAGIC 1. Types of priors - a more informed prior is always better if this information is available
# MAGIC 2. The size of the data or the observations and the value of the theta parameter - more data results in better inference overall, the larger the difference in theta the easier to determine these rates
# MAGIC 3. The number of drawn samples - better and more accurate inference
# MAGIC 4. The number of chains - should reduce variance
# MAGIC 5. The number of cores - cores should be no more than the total number of chains and should be limited to the total number of cores on your hardware, you should see an increase in speed or decrease in runtime as you increase the number of cores.
# MAGIC 
# MAGIC Note that this uses the Metropolis algorithm since this is a discrete sampling problem.

# COMMAND ----------

# ------------ Create the data ---------- #
n_1 = 70
??_real_1 = 7.5
#?? = 0.1
 # Simulate some data
counts_1 = np.random.poisson(??_real_1,n_1)
#plt.bar(np.arange(len(counts_1)),counts_1)

n_2 = 70
??_real_2 = 2.0
#?? = 0.1
 # Simulate some data
counts_2 = np.random.poisson(??_real_2,n_2)
#plt.bar(np.arange(len(counts_2)),counts_2)

total_data = np.concatenate((counts_1, counts_2))
n_counts = len(counts_1) + len(counts_2)
plt.figure()
plt.bar(np.arange(len(total_data)),total_data)

# ------------ Generate the model ----------- #

with pm.Model() as model_poisson:

    alpha_1 = 1.0 / counts_1.mean()
    alpha_2 = 1.0 / counts_2.mean()

    # Different priors have different results                     
    lambda_1 = pm.Exponential("lambda_1", alpha_1)
    lambda_2 = pm.Exponential("lambda_2", alpha_2) 
    #lambda_1 = pm.Gamma("lambda_1", 2, 0.1)
    #lambda_2 = pm.Gamma("lambda_2", 2, 0.1)
    #lambda_1 = pm.Uniform("lambda_1",lower=0, upper=5)
    
    # Uniform prior for the day since we have no information, if we do we should modify the prior to         
    # incorporate that information
    tau = pm.DiscreteUniform("tau", lower=0, upper=n_counts - 1)
    idx = np.arange(n_counts) # id for the day
    lambda_c = pm.math.switch(tau > idx, lambda_1, lambda_2) # switch rate depending on the tau drawn

    observation = pm.Poisson("obs", lambda_c, observed=total_data)
    trace = pm.sample(5000, chains=10, cores=4)

az.plot_trace(trace)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualize \\(\tau\\)
# MAGIC 
# MAGIC We can use the ECDF (Empirical Cumulative Distribution Function) to visualize the distribution of \\(\tau\\). The ECDF helps to visualize the distribution by plotting the CDF as opposed to the binning techniques used by a histogram. It also helps us to identify what values in the data are beneath a certain probability.
# MAGIC 
# MAGIC A reasonable scenario looks like the following:
# MAGIC 
# MAGIC 1. Starting from day 1 till day 'd', it is expected that the rate parameter will be \\(\lambda_1\\), i.e with probability 100%. 
# MAGIC 2. On day 'd', it is possible that the rate is \\(\lambda_1\\) with probability 'x' which implies that the rate could be \\(\lambda_2\\) with probability '1 - x'. So this means that  the distribution of \\(\tau\\) has some probability mass on day 'd' indicating that the rate parameter switches to \\(\lambda_2\\) on this day. 
# MAGIC 3. For days after day 'd', the rate is \\(\lambda_2\\) with probability 100%.

# COMMAND ----------

from statsmodels.distributions.empirical_distribution import ECDF
print('Tau is ',trace['tau'])
print("Length of tau", len(trace['tau']))
print('Lambda 1 is ',trace['lambda_1'])
print("Length of Lambda 1 ",len(trace['lambda_1']))
ecdf = ECDF(trace['tau'])
plt.plot(ecdf.x, ecdf.y, '-')

# COMMAND ----------

for elem in idx:
    prob_lambda_2 = ecdf([elem])
    prob_lambda_1 = 1.0 - prob_lambda_2
    print("Day %d, the probability of rate being lambda_1 is %lf and lambda_2 is %lf "%(elem, prob_lambda_1, prob_lambda_2))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Expected Value of Cases
# MAGIC 
# MAGIC For each draw of $\tau$, there is a draw of $\lambda_1$ and $\lambda_2$. We can use the principles of Monte Carlo approximation to compute the expected value of COVID-19 cases on any day. 
# MAGIC 
# MAGIC Expected value for day = $ \dfrac{1}{N} \sum_{0}^{nsamples}$ Lambda_draw ;  day > Tau_draw ? lambda_2_draw : lambda_1_draw
# MAGIC 
# MAGIC * Draws are in combinations of \\((\lambda_1, \lambda_2, \tau)\\), we want to average out the \\(\lambda\\) value based on the proportion of \\(\lambda\\) suggestions as indicated by the samples
# MAGIC 
# MAGIC * For days 0,...68 we see that the probability of \\(\lambda_1\\) is 1 whereas the probability of \\(\lambda_2\\) is 0. So the expected value is just the average of all the \\(\lambda_1\\) samples.
# MAGIC 
# MAGIC * Similarly, for days from 72,... the probability of \\(\lambda_2\\) is 1 and the probability of \\(\lambda_1\\) is 0. So the expected value of \\(\lambda\\) is just the average of all the \\(\lambda_2\\) samples.
# MAGIC 
# MAGIC * For days in between - let us assume for day 69, we have 10% of the samples indicating that \\(\tau\\) is 70 while 90% indicate that \\(\tau\\) is 69. 
# MAGIC 
# MAGIC  * If \\(\tau\\) is 70, that means that day 69 has rate \\(\lambda_1\\) but if \\(\tau\\) is 69 that implies that day 69 has rate \\(\lambda_2\\). 
# MAGIC        
# MAGIC  * The contribution to the expected value will 10% coming from sum(lambda_1_samples_that_have_tau_70) and 90% coming sum(lambda_2_samples_that_have_tau_69)
# MAGIC               

# COMMAND ----------

print(lambda_1_samples.mean())
print(lambda_2_samples.mean())

# COMMAND ----------

tau_samples = trace['tau']
lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']

N = tau_samples.shape[0]
expected_values = np.zeros(n_counts)
for day in range(0, n_counts):
    # ix is a bool index of all tau samples corresponding to
    # the switchpoint occurring prior to value of 'day'
    ix = day < tau_samples
    # Each posterior sample corresponds to a value for tau.
    # For each day, that value of tau indicates whether we're "before"
    # (in the lambda1 "regime") or
    #  "after" (in the lambda2 "regime") the switchpoint.
    # by taking the posterior sample of lambda1/2 accordingly, we can average
    # over all samples to get an expected value for lambda on that day.
    expected_values[day] = (lambda_1_samples[ix].sum() + lambda_2_samples[~ix].sum()) / N

expected_values

# COMMAND ----------

plt.figure(figsize=(12,8))
plt.bar(np.arange(len(total_data)),total_data)
plt.plot(np.arange(n_counts), expected_values, color='g', lw='4')

# COMMAND ----------

# MAGIC %md
# MAGIC ### GRADED EVALUATION (30 mins)
# MAGIC 
# MAGIC 1. Mean-centering the data helps MCMC Sampling by
# MAGIC 
# MAGIC     a. Reducing the correlation between the variables 
# MAGIC     
# MAGIC     b. Reducing the variance of the variables 
# MAGIC     
# MAGIC     
# MAGIC 2. Hierarchical Linear Regression is beneficial when pooling data results in vital group information being lost
# MAGIC 
# MAGIC     a. True 
# MAGIC     
# MAGIC     b. False 
# MAGIC     
# MAGIC     
# MAGIC 3. Does this code indicate a 
# MAGIC 
# MAGIC     a. Hierarchical model
# MAGIC     
# MAGIC     b. Non-hierarchical model 
# MAGIC     
# MAGIC 
# MAGIC ```
# MAGIC     ??_tmp = pm.Normal('??_tmp', mu=2, sd=5, shape=M)
# MAGIC     ?? = pm.Normal('??', mu=0, sd=10, shape=M)
# MAGIC     ?? = pm.HalfCauchy('??', 5)
# MAGIC     ?? = pm.Exponential('??', 1/30)
# MAGIC     
# MAGIC     y_pred = pm.StudentT('y_pred', mu=??_tmp[idx] + ??[idx] * x_centered,
# MAGIC                          sd=??, nu=??, observed=y_m)
# MAGIC    
# MAGIC ```
# MAGIC 
# MAGIC     
# MAGIC   
# MAGIC 4. Non-hierarchical Linear Regression with groups of sparse data can result in
# MAGIC 
# MAGIC     a. very large credible intervals 
# MAGIC     
# MAGIC     b. very small credible intervals
# MAGIC     
# MAGIC 
# MAGIC 5. In Hierarchical Linear Regression, priors (not hyperpriors) on variable distributions 
# MAGIC 
# MAGIC     a. are constant values
# MAGIC     
# MAGIC     b. distributions
# MAGIC     
# MAGIC     
# MAGIC 6. Polynomial Regression is useful for
# MAGIC 
# MAGIC     a. Linear data
# MAGIC     
# MAGIC     b. Non-linear data
# MAGIC     
# MAGIC     
# MAGIC 7. Multiple Linear Regression is used when you have
# MAGIC 
# MAGIC     a. Multiple dependent or target variables
# MAGIC     
# MAGIC     b. Multiple independent or predictor variables
# MAGIC     
# MAGIC     
# MAGIC 8. PyMC3 allows you to model multiple predictor variables uing the shape parameter in a distribution without having to create multiple parameters explicitly
# MAGIC 
# MAGIC     a. True 
# MAGIC     
# MAGIC     b. False
# MAGIC     
# MAGIC 
# MAGIC 9. The inverse link function of linear regression is
# MAGIC 
# MAGIC     a. Logit function
# MAGIC     
# MAGIC     b. Identity function
# MAGIC    
# MAGIC    
# MAGIC 10. Multiclass classification uses the
# MAGIC 
# MAGIC     a. Sigmoid function 
# MAGIC     
# MAGIC     b. Softmax function
# MAGIC     
# MAGIC     
# MAGIC 11. A binary classification problem uses the Bernoulli distribution to model the target, the multiclass classification uses
# MAGIC 
# MAGIC     a. Categorical distribution
# MAGIC     
# MAGIC     b. Poisson distribution

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC <br>
# MAGIC <hr style="border:2px solid blue"> </hr>
# MAGIC 
# MAGIC 
# MAGIC ### MCMC Metrics
# MAGIC 
# MAGIC #### Paper discussing Bayesian visualization
# MAGIC 
# MAGIC https://arxiv.org/pdf/1709.01449.pdf
# MAGIC 
# MAGIC #### Tuning
# MAGIC 
# MAGIC [Colin Caroll's talk](https://colcarroll.github.io/hmc_tuning_talk/)
# MAGIC 
# MAGIC When a step size is required, PyMC3 uses the first 500 steps varying the step size to get to an acceptance rate of 23.4%. 
# MAGIC 
# MAGIC * The acceptance rate is the proportion of proposed values that are not rejected during the sampling process. Refer to Course 2, Metropolis Algorithms for more information. 
# MAGIC 
# MAGIC * In Metropolis algorithms, the step size is related to the variance of the proposal distribution. See the next section for more information.
# MAGIC 
# MAGIC These are the default numbers that PyMC3 uses, which can be modified. It was reported in a study that the acceptance rate of 23.4% results in the highest efficiency for Metropolis Hastings. These are empirical results and therefore should be treated as guidelines. According to the SAS Institute, a high acceptance rate (90% or so) usually is a sign that the new samples are being drawn from points close to the existing point and therefore the sampler is not exploring the space much. On the other hand, a low acceptance rate is probably due to inappropriate proposal distribution causing new samples to be rejected. PyMC3 aims to get an acceptance rate between 20% and 50% for Metropolis Hastings, 65% for Hamiltonian Monte Carlo (HMC) and 85% for No U-Turn Sampler (NUTS)
# MAGIC 
# MAGIC If you have convergence issues as indicated by the visual inspection of the trace, you can try increasing the number of samples used for tuning. It is also worth pointing out that there is more than just step-size adaptation that is happening during this tuning phase. 
# MAGIC 
# MAGIC `pm.sample(num_samples, n_tune=num_tuning)`
# MAGIC 
# MAGIC ##### Metropolis algorithm
# MAGIC In the Metropolis algorithm the standard deviation of the proposal distribution is a tuning parameter that can be set while initializing the algorithm. 
# MAGIC 
# MAGIC $x_{t+1} \sim Normal(x_t, stepsize \cdot I)$
# MAGIC 
# MAGIC The larger this value, the larger the space from where new samples can be drawn. If the acceptance rate is too high, increase this standard deviation. Keep in mind that you run the risk of getting invalid draws if it is too large. 
# MAGIC 
# MAGIC ##### Hamiltonian Monte Carlo (HMC) algorithm
# MAGIC 
# MAGIC The HMC algorithm is based on the solution of differential equations known as Hamilton's equations. These differential equations depend on the probability distributions we are trying to learn. We navigate these distributions by moving around them in a trajectory using steps that are defined by a position and momentum at that position. Navigating these trajectories can be a very expensive process and the goal is to minimize this computational effort in this process.
# MAGIC 
# MAGIC To get a sense of the intuition behind HMC, it is based on the notion of conservation of energy. When the sampler trajectory is far away from the probability mass center, it has high potential energy but low kinetic energy and when it is closer to the center of the probability mass, it will have high kinetic energy but low potential energy.
# MAGIC 
# MAGIC The step size, in HMC, corresponds to the covariance of the momentum distribution that is sampled. Smaller step sizes move slowly in the manifold, however larger step sizes can result in integration errors. There is a Metropolis step at the end of the HMC algorithm and the  target acceptance rates of 65% in PyMC3 corresponds to this Metropolis step.
# MAGIC 
# MAGIC #### Mixing 
# MAGIC 
# MAGIC Mixing refers to how well the sampler covers the 'support' of the posterior distribution or rather how well it covers the entire distribution. Poor convergence is often a result of poor mixing. This can happen due to the choice of 
# MAGIC 
# MAGIC 1. The choice of an inappropriate proposal distribution for Metropolis 
# MAGIC 2. If we have too many correlated variables
# MAGIC 
# MAGIC The underlying cause for this can be
# MAGIC 
# MAGIC 1. Too large a step size
# MAGIC 2. Not running the sampler long enough
# MAGIC 3. Multimodal distributions 
# MAGIC 
# MAGIC 
# MAGIC #### Rhat
# MAGIC 
# MAGIC We can compute a metric called Rhat (also called the potential scale reduction factor) that measures the ratio of the variance between the chains to the variance within the chains. It is calculated as the ratio of the standard deviation using the samples from all the chains (all samples appended together from each chain) over the  RMS of the within-chain standard deviations of all the chains. Poorly mixed samples will have greater variance in the accumulated samples (numerator) compared to the variance in the individual chains. It was empirically determined that Rhat values below 1.1 are considered acceptable while those above it are indications of a lack of convergence in the chains. Gelman et al. (2013) introduced a split Rhat that compares the first half with the second half of the samples from each chain to improve upon the regular Rhat. Arviz implements a split Rhat as can be seen from [Arviz Rhat](https://arviz-devs.github.io/arviz/generated/arviz.rhat.html). There is also an improved rank-based Rhat
# MAGIC [Improved Rhat](https://arxiv.org/pdf/1903.08008.pdf).
# MAGIC 
# MAGIC `az.rhat(trace, method='split')`
# MAGIC 
# MAGIC ### Centered vs. Non-centered Parameterization
# MAGIC 
# MAGIC [T Wiecki on Reparametrization](https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/)
# MAGIC 
# MAGIC When there is insufficient data in a hierarchical model, the variables being inferred end up having correlation effects, thereby making it difficult to sample. One obvious solution is to obtain more data, but when this isn't possible we resort to reparameterization by creating a non-centered model from the centered model.
# MAGIC 
# MAGIC #### Centered Model
# MAGIC 
# MAGIC $$ \mu \sim Normal (0,1)$$
# MAGIC 
# MAGIC $$ \sigma \sim HalfNormal(1)$$
# MAGIC 
# MAGIC $$y_i \sim Normal(\mu, \sigma)$$
# MAGIC 
# MAGIC And we try to fit the two parameters for $\mu$ and $\sigma$ directly here.
# MAGIC 
# MAGIC #### Non-centered Model
# MAGIC 
# MAGIC 
# MAGIC $$ \mu \sim Normal (0,1)$$
# MAGIC 
# MAGIC $$ \sigma \sim HalfNormal(1)$$
# MAGIC 
# MAGIC $$y_{i\_unit} \sim Normal(0,1)$$
# MAGIC 
# MAGIC $$y = \mu + \sigma y_{i\_unit}$$

# COMMAND ----------

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import norm, halfcauchy, halfnorm


def centered_model():

    # generate data
    np.random.seed(0)
    n = 1
    m = 10000
    mu = norm.rvs(0, 3, m)
    sigma = halfnorm.rvs(0, 2, m)
    y = norm.rvs(mu, sigma, (n, m))

    # set up model
    with pm.Model():

        mu_ = pm.Normal("mu", 0, 1)
        sigma_ = pm.HalfNormal("sigma", 1)
        y_ = pm.Normal("y", mu_, sigma_, shape=n)

        # sample and save samples
        trace = pm.sample(m, chains=3)
        mu_samples = trace["mu"][:]
        sigma_samples = trace["sigma"][:]
        y_samples = trace["y"].T[:]
        
    sc = 5
    fig, axes = plt.subplots(2, 2, constrained_layout=False, sharex=True)
    ax = axes[0, 0]
    ax.scatter(y[0], mu, marker=".", alpha=0.05, rasterized=True)
    ax.set_xlim(-sc, sc)
    ax.set_ylim(-sc, sc)
    ax.set_ylabel("true $\mu$")
    ax.set_xlabel("true $y$")

    ax = axes[0, 1]
    ax.scatter(y_samples[0], mu_samples, marker=".", alpha=0.05, rasterized=True, color="r")
    ax.set_ylim(-sc, sc)
    ax.set_xlim(-sc, sc)
    ax.set_yticklabels([])
    ax.set_ylabel("$\mu$ samples")
    ax.set_xlabel("y samples")

    ax = axes[1, 0]
    ax.scatter(y[0], sigma, marker=".", alpha=0.05, rasterized=True)
    ax.set_ylim(0, sc / 2)
    ax.set_ylabel("true $\sigma$")
    ax.set_xlabel("true y")
    
    ax = axes[1, 1]
    ax.scatter(y_samples[0], sigma_samples, marker=".", alpha=0.05, rasterized=True, color="r")
    ax.set_ylim(0, sc / 2)
    ax.set_yticklabels([])
    ax.set_ylabel("$\sigma$ samples")
    ax.set_xlabel("y samples")

    plt.show()
    return(trace)
    

    
def noncentered_model():

    # generate data
    np.random.seed(0)
    n = 1
    m = 10000
    mu = norm.rvs(0, 3, m)
    sigma = halfnorm.rvs(0, 2, m)
    y = norm.rvs(mu, sigma, (n, m))

    # set up model
    with pm.Model():

        mu_ = pm.Normal("mu", 0, 1)
        sigma_ = pm.HalfNormal("sigma", 1)
        yt_ = pm.Normal("yt", 0, 1, shape=n)
        pm.Deterministic("y", mu_ + yt_ * sigma_)
        # y_ = pm.Normal("y", mu_, sigma_, shape=n)

        # sample and save samples
        trace = pm.sample(m, chains=3)
        mu_samples = trace["mu"][:]
        sigma_samples = trace["sigma"][:]
        yt_samples = trace["yt"].T[:]
        y_samples = trace["y"].T[:]

    # plot 2-D figures
    sc = 5
    fig, axes = plt.subplots(2, 2, constrained_layout=False, sharex=True)
    
    ax = axes[0, 0]
    ax.scatter(yt_samples[0], mu_samples, marker=".", alpha=0.05, rasterized=True, color="salmon")
    ax.set_xlim(-sc, sc)
    ax.set_ylim(-sc, sc)
    ax.set_ylabel("$\mu$ samples")
    ax.set_xlabel("ncm - y unit Normal samples")
    ax.set_xticklabels([])

    ax = axes[0, 1]
    ax.scatter(y_samples[0], mu_samples, marker=".", alpha=0.05, rasterized=True, color="r")
    ax.set_xlim(-sc, sc)
    ax.set_ylim(-sc, sc)
    ax.set_ylabel("$\mu$ samples")
    ax.set_xlabel("ncm - y samples")
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax = axes[1, 0]
    ax.scatter(yt_samples[0], sigma_samples, marker=".", alpha=0.05, rasterized=True, color="salmon")
    ax.set_xlim(-sc, sc)
    ax.set_ylim(0, sc / 2)
    ax.set_xlabel("ncm - y unit Normal samples")
    ax.set_ylabel("$\sigma$ samples")

    ax = axes[1, 1]
    ax.scatter(y_samples[0], sigma_samples, marker=".", alpha=0.05, rasterized=True, color="r")
    ax.set_xlim(-sc, sc)
    ax.set_ylim(0, sc / 2)
    ax.set_yticklabels([])
    ax.set_xlabel("ncm - y samples")
    ax.set_ylabel("$\sigma$ samples")
    
    plt.show()
    return(trace)


trace_cm = centered_model()
trace_ncm = noncentered_model()

# COMMAND ----------

plt.figure()
plt.scatter(trace_ncm['mu'], trace_ncm['sigma'],c='teal', alpha=0.1)
plt.scatter(trace_cm['mu'], trace_cm['sigma'], c='yellow', alpha=0.1)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Convergence

# COMMAND ----------

import arviz as az
print("------------ Centered model ------------")

# The bars indicate the location of the divergences in the sampling process
az.plot_trace(trace_cm, divergences='bottom')
az.summary(trace_cm)

# COMMAND ----------

print("------------ Non-centered model ------------")

# The bars indicate the locations of divergences in the sampling process
az.plot_trace(trace_ncm, divergences='top')
az.summary(trace_ncm)

# COMMAND ----------

# MAGIC %md
# MAGIC 1. The posterior densities have more agreement for the non-centered model(ncm) compared to the centered model (cm), for the different chains.
# MAGIC 2. There are more divergences for centered model compared to the non-centered model as can be seen from the vertical bars in the trace plot. 
# MAGIC 3. In general, the non-centered model mixes better than the centered model - non-centered model looks fairly evenly mixed while centered model looks patchy in certain regions. 
# MAGIC 4. It is possible to see flat lines in the trace for a centered model, a flat line indicates that the same sample value is being used because all new proposed samples are being rejected, in other words the sampler is sampling slowly and not getting to a different space in the manifold. The only fix here is to sample for longer periods of time, however we are assuming that we can get more unbiased samples if we let it run longer.
# MAGIC 
# MAGIC 
# MAGIC #### Forest Plot
# MAGIC 
# MAGIC We plot the densities of both the cm and the ncm models, notice the differences in effective sample sizes for the centered model (very low).

# COMMAND ----------

fig, axs = plt.subplots(1,3)
fig.set_size_inches(18.5, 10.5)
az.plot_forest([trace_cm, trace_ncm], var_names=['sigma'], 
               kind = 'ridgeplot',
               model_names=['Centered','Non-centered'],
               combined=False, 
               ess=True, 
               r_hat=True, 
               ax=axs[0:3], 
               figsize=(20,20) )
#az.plot_forest(trace_ncm, var_names=['a'],
#               kind='ridgeplot',
#               combined=False, 
#               ess=True, 
#               r_hat=True, 
#               ax=axs[1,0:3])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Autocorrelation and effective sample sizes
# MAGIC 
# MAGIC Ideally, we would like to have zero correlation in the samples that are drawn. Correlated samples violate our condition of independence and can give us biased posterior estimates of our posterior distribution. 
# MAGIC Thinning or pruning refers to the process of dropping every nth sample from a chain. This is to minimize the number of correlated samples that might be drawn, especially if the proposal distribution is narrow. The autocorrelation plot computes the correlation of a sequence with itself but shifted by n; for each n on the x axis the corresponding value of autocorrelation is plotted on the y axis.  
# MAGIC 
# MAGIC `az.plot_autocorr(trace, var_names=["a", "b"])`
# MAGIC 
# MAGIC Techniques like Metropolis-Hastings are susceptible to having auto-correlated samples. We plot the autocorrelation here for the cm and the ncm models. The cm models have samples that have a high degree of autocorrelation while the ncm models does not.

# COMMAND ----------

fig, axs = plt.subplots(3,2)
fig.set_size_inches(12, 18)
az.plot_autocorr(trace_cm, var_names=['sigma'], ax=axs[0:3,0])
az.plot_autocorr(trace_ncm, var_names=['sigma'], ax=axs[0:3,1])
axs[0][0].set_title('Sigma - centered model')
axs[0][1].set_title('Sigma - non-centered model')

# COMMAND ----------

# MAGIC %md
# MAGIC Since a chain with autocorrelation has fewer samples that are independent, we can calculate the number of effective samples called the effective sample size. This is listed when a summary of the trace is printed out, however it can also be explicitly computed using
# MAGIC 
# MAGIC `az.effective_n(trace_s)`
# MAGIC 
# MAGIC PyMC3 will throw a warning if the number of effective samples is less than 200 (200 is heuristically determined to provide a good approximation for the mean of a distribution). Unless you want to sample from the tails of a distribution (rare events), 1000 to 2000 samples should provide a good approximation for a distribution.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC #### Monte Carlo error 
# MAGIC 
# MAGIC The Monte Carlo error is a measure of the error of our sampler which stems from the fact that not all samples that we have drawn are independent. This error is defined by dividing a trace into 'n' blocks. We then compute the mean of these blocks and calculate the error as the standard deviation of these means over the square root of the number of blocks.
# MAGIC 
# MAGIC $mc_{error} = \sigma(\mu(block_i)) / \sqrt(n)$
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC #### Divergence
# MAGIC 
# MAGIC Divergences happen in regions of high curvature or high gradient in the manifold. When PyMC3 detects a divergence it abandons that chain, and as a result the samples that are reported to have been diverging are close to the space of high curvature but not necessarily right on it.
# MAGIC 
# MAGIC In some cases, PyMC3 can indicate falsely that some samples are divergences, this is due to the heuristics used to identify divergences. Concentration of samples in a region is an indication that these are not divergences. 
# MAGIC 
# MAGIC We visualize this for the cm and ncm models with pairplots of the variables. You can see how the cm models have difficulty sampling at the edge of the funnel shaped two-dimensional manifold formed by the pairplot.

# COMMAND ----------

# Get the divergences
print("Number of divergences in cm model, %d and %lf percent " % (trace_cm['diverging'].nonzero()[0].shape[0], trace_cm['diverging'].nonzero()[0].shape[0]/ len(trace_cm) * 100))
divergent = trace_cm['diverging']

print("Number of divergences in ncm model, %d and %lf percent " % (trace_ncm['diverging'].nonzero()[0].shape[0], trace_ncm['diverging'].nonzero()[0].shape[0]/ len(trace_ncm) * 100))
divergent = trace_cm['diverging']

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pairplot

# COMMAND ----------

print("Centered model")
az.plot_pair(trace_cm, var_names = ['mu', 'sigma', 'y'], divergences=True)

# COMMAND ----------

print("Non-centered model")
az.plot_pair(trace_ncm, var_names = ['mu', 'sigma', 'y'], divergences=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Parallel Coordinates
# MAGIC 
# MAGIC You can also have a parallel coordinates plot of the variables to look at the multidimensional data instead of pairplots. If we notice tight-knit lines around a region, that is an indication of difficulty sampling and hence divergences. This behavior can be observed in the centered model around 0 while the non-centered model has a sparser cluster of lines around 0. Sparser clusters can be an indication of false positives where divergences are reported. Apart from reformulating the problem, there are two ways to avoid the problem of divergences. 
# MAGIC 
# MAGIC 1. Increase the tuning samples
# MAGIC 2. Increase 'target_accept'
# MAGIC 
# MAGIC The parallel coordinates below show a much denser set of lines for the divergences for the centered model.

# COMMAND ----------

fig, axs = plt.subplots(2,1)
fig.set_size_inches(20,20)
axs[0].set_title('CM model')
axs[1].set_title('NCM model')
az.plot_parallel(trace_cm, var_names=['mu','sigma','y'], figsize=(20,20), shadend=0.01, colord='tab:blue', textsize=15, ax=axs[0])
az.plot_parallel(trace_ncm, var_names=['mu','sigma','y'], figsize=(20,20), shadend=0.01, colord='tab:blue', textsize=15,ax=axs[1])

# COMMAND ----------

# MAGIC %md
# MAGIC #### A note on why we compute the log of the posterior
# MAGIC 
# MAGIC In short, this is done to avoid numerical overflow or underflow issues. When dealing with really large or small numbers, it is likely the limited precision of storage types (float, double etc.) can be an issue. In order to avoid this, the log of the probabilities are used instead in the calculations.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Revisiting the Multiclass Classification Problem

# COMMAND ----------

y_s = iris_data.target
x_n = iris_data.columns[:-1]
x_s = iris_data[x_n]
x_s = (x_s - x_s.mean()) / x_s.std()
x_s = x_s.values

import theano as tt
#tt.config.gcc.cxxflags = "-Wno-c++11-narrowing"

with pm.Model() as model_mclass:
    alpha = pm.Normal('alpha', mu=0, sd=5, shape=3)
    beta = pm.Normal('beta', mu=0, sd=5, shape=(4,3))
    ?? = pm.Deterministic('??', alpha + pm.math.dot(x_s, beta))
    ?? = tt.tensor.nnet.softmax(??)
    yl = pm.Categorical('yl', p=??, observed=y_s)
    trace_s = pm.sample(2000)

data_pred = trace_s['??'].mean(0)
y_pred = [np.exp(point)/np.sum(np.exp(point), axis=0) for point in data_pred]
az.plot_trace(trace_s, var_names=['alpha'])
f'{np.sum(y_s == np.argmax(y_pred, axis=1)) / len(y_s):.2f}'

# COMMAND ----------

# MAGIC %md
# MAGIC #### Diagnostics
# MAGIC 
# MAGIC All diagnostics in PyMC3 are now in Arviz starting with version 3.9 of PyMC3. The 'summary' method is a good place to start.

# COMMAND ----------

az.summary(trace_s)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Rhat

# COMMAND ----------

az.rhat(trace_s)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Stat Names
# MAGIC 
# MAGIC Print out the available statistics for your model.

# COMMAND ----------

trace_s.stat_names

# COMMAND ----------

import seaborn as sns
print("Length of trace_s",len(trace_s.energy), max(trace_s.energy))
print("Depth of the tree used to generate the sample  ",trace_s.depth)
# If tree size is too large, it is an indication of difficulty sampling, due to correlations, sharp posterior space
# or long-tailed posteriors. A solution is to reparameterize the model.
print("Tree size  ",trace_s.tree_size)
print("Energy at the point where the sample was generated ",trace_s.energy)
# This is difference in energy beween the start and the end of the trajectory, should ideally be zero
print("Energy error between start and end of the trajectory ",trace_s.energy_error)
# maximum difference in energy along the whole trajectory of sampling, this can help identify divergences
print("Energy error maximum over the entire trajectory ",trace_s.max_energy_error)
print("Step size ",trace_s.step_size)
print("Best step size determined from tuning ",trace_s.step_size_bar)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Trace Energy
# MAGIC 
# MAGIC Ideally, you want the energy of your trace and the transition energy to be similar. If your transition energy is too narrow, it could imply that your sampler does not have enough energy to sample the entire posterior space and the sampled results may not appropriately represent the posterior well (biased estimate).

# COMMAND ----------

energy_diff = np.diff(trace_s.energy)
sns.distplot(trace_s.energy - trace_s.energy.mean(), label="Energy of trace")
sns.distplot(energy_diff, label="Transition energy")
plt.legend()

# COMMAND ----------

# Seaborn uses the interquartile range to draw the box plot whiskers given by Q1 - 1.5IQR, Q3 + 1.5IQR
# The boxplot helps to better visualize the density of outliers
sns.boxplot(trace_s.max_energy_error)

# COMMAND ----------

# MAGIC %md
# MAGIC The energy and the energy transition should be as close as possible if the energy transition is smaller or narrower than the marginal energy, it implies that the sampler did not sample the space appropriately and that the results obtained are probably biased.

# COMMAND ----------

pm.energyplot(trace_s)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step size
# MAGIC 
# MAGIC The variation of step size through the sampling process.

# COMMAND ----------

sns.lineplot(np.arange(0,len(trace_s.step_size_bar)), trace_s.step_size_bar)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Convergence with the Geweke Score
# MAGIC 
# MAGIC The Geweke score is a z-score that is computed at various segments in the time-series for an inferred parameter. 
# MAGIC A score of less than 2 (less than 2 standard deviations) indicates good convergence. It computes the z-score between each segment and the last 50%, by default, from the sampled chain. The function `pm.geweke` returns an array of (interval start location, z-score)

# COMMAND ----------

pm.geweke(trace_s['alpha'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Divergences

# COMMAND ----------

# Get the divergences
print("Number of divergences %d and percent %lf " % (trace_s['diverging'].nonzero()[0].shape[0], trace_s['diverging'].nonzero()[0].shape[0]/ len(trace_s) * 100))
divergent = trace_s['diverging']
beta_divergent = trace_s['beta'][divergent]
print("Shape of beta_divergent - Sum of divergences from all chains x shape of variable ", beta_divergent.shape)

# COMMAND ----------

import pprint
print("Total number of warnings ",len(trace_s.report._warnings))
pprint.pprint(trace_s.report._warnings[0])

dir(trace_s.report._warnings[0])

print("---------- Message ----------")
pprint.pprint(trace_s.report._warnings[0].message)
print("---------- Kind of warning ----------")
pprint.pprint(trace_s.report._warnings[0].kind)
print("---------- Level ----------")
pprint.pprint(trace_s.report._warnings[0].level)
print("---------- Step ----------")
pprint.pprint(trace_s.report._warnings[0].step)
print("---------- Source ---------- ")
pprint.pprint(trace_s.report._warnings[0].divergence_point_source)
print("---------- Destination ---------- ")
pprint.pprint(trace_s.report._warnings[0].divergence_point_dest)

# COMMAND ----------

trace_s.report._warnings[0].divergence_info

# COMMAND ----------

for elem in trace_s.report._warnings:
    print(elem.step)

# COMMAND ----------

import theano as tt 
import arviz as az 

# If we run into the identifiability problem, we can solve for n-1 variables 
with pm.Model() as model_sf:
    ?? = pm.Normal('??', mu=0, sd=2, shape=2)
    ?? = pm.Normal('??', mu=0, sd=2, shape=(4,2))
    ??_f = tt.tensor.concatenate([[0] ,??])
    ??_f = tt.tensor.concatenate([np.zeros((4,1)) , ??], axis=1)
    ?? = ??_f + pm.math.dot(x_s, ??_f)
    ?? = tt.tensor.nnet.softmax(??)
    yl = pm.Categorical('yl', p=??, observed=y_s)
    trace_sf = pm.sample(1000)
    
az.plot_trace(trace_sf, var_names=['??'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Diagnosing MCMC using PyMC3
# MAGIC 
# MAGIC It is a good idea to inspect the quality of the solutions obtained. It is possible that one obtains suboptimal samples resulting in biased estimates, or the sampling is slow. There are two broad categories of tests, a visual inspection and a quantitative assessment. There are a few things that can be done if one suspects sampling issues.
# MAGIC 
# MAGIC 1. More samples, it is possible that there aren't sufficient samples to come up with an appropriate posterior.
# MAGIC 2. Use burn-in, this is removing a certain number of samples from the beginning while PyMC3 is figuring out the step size. This is set to 500 by default. With tuning it is not necessary to explicitly get rid of samples from the beginning.
# MAGIC 3. Increase the number of samples used for tuning.
# MAGIC 4. Increase the target_accept parameter as 
# MAGIC 
# MAGIC     `pm.sample(5000, chains=2, target_accept=0.95)`
# MAGIC     
# MAGIC     `pm.sample(5000, chains=2, nuts_kwargs=dict(target_accept=0.95))`
# MAGIC     
# MAGIC     Target_accept is the acceptance probability of the samples. This has the effect of varying the step size in the MCMC process so that we get the desired acceptance probability as indicated by the value of target_accept. It is a good idea to take smaller steps especially during Hamiltonian Monte Carlo so as to explore regions of high curvature better. Smaller step sizes lead to larger acceptance rates and larger step sizes lead to smaller acceptance rates. If the current acceptance rate is smaller than the target acceptance rate, the step size is reduced to increase the current acceptance rates.
# MAGIC     
# MAGIC 5. Reparameterize the model so that the model, while remaining the same, is expressed differently so that is easier for the sampler to explore the distribution space.
# MAGIC 6. Modify the data representation - mean centering and standardizing the data are two standard techniques that can be applied here. Note that (5) refers to model transformation while this is data transformation.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Debugging PyMC3

# COMMAND ----------

x = np.random.randn(100)

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sigma=1)
    sd = pm.Normal('sd', mu=0, sigma=1)
    
    mu_print = tt.printing.Print('mu')(mu)
    sd_print = tt.printing.Print('sd')(sd)

    obs = pm.Normal('obs', mu=mu_print, sigma=sd_print, observed=x)
    step = pm.Metropolis()
    trace = pm.sample(5, step)
    
trace['mu']

# COMMAND ----------

# MAGIC %md
# MAGIC ### Arviz Data Representation
# MAGIC 
# MAGIC From the Arviz page, it states that, apart from NumPy arrays and Python dictionaries, there is support for a few data structures such as xarrays, InferenceData and NetCDF. While NumPy and dictionaries are great for in-memory computations, the other file formats are suitable for persisting computed data and models to disk. InferenceData is a high-level data structure that holds the data in a storage format such as NetCDF.
# MAGIC 
# MAGIC [Xarray documentation](http://xarray.pydata.org/en/stable/why-xarray.html)
# MAGIC 
# MAGIC [NetCDF documentation](http://unidata.github.io/netcdf4-python/netCDF4/index.html)
# MAGIC 
# MAGIC ![Structure](https://arviz-devs.github.io/arviz/_images/InferenceDataStructure.png)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load the school data

# COMMAND ----------

data = az.load_arviz_data("centered_eight")
data

# COMMAND ----------

data.posterior.get('mu')

# COMMAND ----------

data = az.load_arviz_data("non_centered_eight")
data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load the S&P500 returns data

# COMMAND ----------

sp500 = pd.read_csv(pm.get_data('SP500.csv'), index_col='Date')
sp500

# COMMAND ----------

# MAGIC %md
# MAGIC ### GRADED EVALUATION (36 min)
# MAGIC 
# MAGIC 1. For Metropolis-Hastings algorithms, an acceptance rate of 23.4 was shown to be ideal
# MAGIC 
# MAGIC     a. True 
# MAGIC     
# MAGIC     b. False
# MAGIC     
# MAGIC 
# MAGIC 2. A high acceptance rate (>90%) is an indication that the sampler is not exploring the space very well
# MAGIC 
# MAGIC     a. True 
# MAGIC     
# MAGIC     b. False
# MAGIC     
# MAGIC     
# MAGIC 3. A low acceptance rate is an indication that 
# MAGIC 
# MAGIC     a. An incorrect proposal distribution is being used 
# MAGIC     
# MAGIC     b. The variance of the proposal distribution is too low
# MAGIC     
# MAGIC   
# MAGIC 4. When using the NUTS algorithm, PyMC3 aims to get an acceptance rate of 
# MAGIC 
# MAGIC     a. 75%
# MAGIC     
# MAGIC     b. 85%
# MAGIC     
# MAGIC     
# MAGIC 5. If you have convergence issues, it is better to
# MAGIC 
# MAGIC     a. Try increasing the total number of samples drawn
# MAGIC     
# MAGIC     b. Try increasing the number of tuning samples
# MAGIC     
# MAGIC     
# MAGIC 6. A step size that is too large can result in 
# MAGIC 
# MAGIC     a. Large sample values
# MAGIC     
# MAGIC     b. Invalid sample values
# MAGIC     
# MAGIC     
# MAGIC 7. Large step sizes in Hamiltonian Monte Carlo can result in 
# MAGIC 
# MAGIC     a. Integration errors
# MAGIC     
# MAGIC     b. Out-of-bounds errors
# MAGIC     
# MAGIC     
# MAGIC 8. Mixing in MCMC refers to
# MAGIC 
# MAGIC     a. How well the sampling covers the entire distribution space
# MAGIC     
# MAGIC     b. The similarity of the sample values
# MAGIC     
# MAGIC     
# MAGIC 9. Rhat, used to measure mixing measures 
# MAGIC 
# MAGIC     a. the variance between the chains
# MAGIC     
# MAGIC     b. the ratio of the variance between the chains to the variance within the chains 
# MAGIC     
# MAGIC     
# MAGIC 10. Rhat values below 1.1 indicate convergence while those above do not
# MAGIC 
# MAGIC     a. True
# MAGIC     
# MAGIC     b. False
# MAGIC     
# MAGIC     
# MAGIC 11. Thinning or pruning refers to dropping every n'th sample to avoid correlated samples
# MAGIC 
# MAGIC     a. True
# MAGIC     
# MAGIC     b. False
# MAGIC     
# MAGIC     
# MAGIC 12. Divergences happen in regions of high curvature or sharp gradients in the sampling manifold
# MAGIC 
# MAGIC     a. True
# MAGIC     
# MAGIC     b. False

# COMMAND ----------


