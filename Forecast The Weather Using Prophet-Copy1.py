#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
weather = pd.read_csv("weather.csv" ,index_col="DATE")


# In[2]:


weather


# In[3]:


null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]


# In[4]:


null_pct


# In[5]:


valid_columns = weather.columns[null_pct <.05]


# In[6]:


valid_columns


# In[7]:


weather = weather[valid_columns].copy()


# In[8]:


weather


# In[9]:


weather.columns = weather.columns.str.lower()


# In[10]:


weather


# In[11]:


weather.index = pd.to_datetime(weather.index)


# In[12]:


weather.index


# In[13]:


weather["station"].unique()


# In[14]:


lga = weather[weather["station"] == "USW00014732"].copy()


# In[15]:


lga


# In[16]:


weather = weather[weather["station"] == "USW00094789"].copy()


# In[17]:


weather


# In[18]:


weather = weather.merge(lga, left_index=True, right_index=True)


# In[19]:


weather


# In[20]:


weather["y"] = weather.shift(-1)["tmax_x"]


# In[21]:


weather[["tmax_x","y"]]


# In[22]:


weather = weather.ffill()


# In[23]:


weather


# In[24]:


weather["ds"] = weather.index


# In[25]:


weather


# In[26]:


predictors = weather.columns[~weather.columns.isin(["y", "name_x", "station_x", "name_y","station_y", "ds"])]


# In[27]:


predictors


# In[28]:


train = weather[:"2021-12-31"]
test = weather["2021-12-31":]


# In[29]:


from prophet import Prophet

def fit_prophet(tarin):
    m = Prophet()
    for p in predictors:
        m.add_regressor(p)
    m.fit(tarin)
    return m
          
m = fit_prophet(train)


# In[30]:


predictions = m.predict(test)


# In[31]:


predictions


# In[32]:


weather.plot("ds", "y")


# In[33]:


from prophet.plot import plot_plotly, plot_components_plotly, plot_cross_validation_metric

plot_components_plotly(m, predictions)


# In[34]:


from prophet.utilities import regressor_coefficients

regressor_coefficients(m)


# In[35]:


predictions.index = test.index
predictions["actual"] = test["y"]


# In[36]:


def mse(predictions, actual_label = "actual", pred_label="yhat"):
   se = ((predictions[actual_label] - predictions[pred_label]) ** 2)
   print(se.mean())

mse(predictions)


# In[37]:


from prophet.diagnostics import cross_validation

m = fit_prophet(weather)
cv = cross_validation(m, initial=f'{365*5}days',period= '180 days', horizon='180 days', parallel="processes")


# In[38]:


mse(cv, actual_label="y")


# In[39]:


cv[["y", "yhat"]] [-365:].plot()


# In[40]:


m = fit_prophet(weather)
m.predict(weather.iloc[-1:])


# In[41]:


m = Prophet()
m.fit(weather)
future = m.make_future_dataframe(periods=365)


# In[42]:


forecast = m.predict(future)


# In[43]:


plot_plotly(m, forecast)


# In[ ]:




