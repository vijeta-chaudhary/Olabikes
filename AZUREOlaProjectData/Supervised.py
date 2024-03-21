#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[11]:


pip install gpxpy


# In[15]:


pip install xgboost


# In[16]:


from sklearn.metrics import mean_squared_error
from math import sqrt, ceil, floor
import matplotlib.pyplot as plt
from xgboost import plot_importance
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from joblib import dump, load
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv('azureml://subscriptions/c9bed033-5af9-4b2f-87bb-5448c9b743aa/resourcegroups/OLA/workspaces/ola/datastores/ola/paths/part-00000-tid-957313629577908374-710d5899-7f56-452e-aa27-061a700b1b91-264-1-c000.csv')


# In[7]:


df.head(10)


# In[9]:


df.shape


# In[8]:


df.dtypes


# In[10]:


df.isnull().sum()


# Based on pickup cluster adding all the ride request that happend in the interval of 30mins for that particular cluster
# 

# all the minutes in the dataset will be rounded of to nearest number in the interval of 30

# In[17]:


def round_timestamp_30interval(x):
    if type(x)==str:
        x = datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')  # Modified format string
    return x- timedelta(minutes=x.minute%30, seconds=x.second, microseconds=x.microsecond)


# In[18]:


df['ts'] = np.vectorize(round_timestamp_30interval)(df['ts'])


# In[19]:


df.head(10)


# - Making a copy of dataframe 'df' 
# - Converting column 'ts' to datetime format (it was in object type before)

# In[20]:


dataset = deepcopy(df)
dataset.ts = pd.to_datetime(dataset.ts)
dataset


# In[21]:


dataset = dataset[['ts','number','pickup_cluster']]


# In[22]:


dataset.head()


# - grouping dataframe by columns ts and pickup_cluster and calculating the unique ride request based on those two columns 
# - 2nd line saves the count of request

# In[23]:


dataset=dataset.groupby(by = ['ts','pickup_cluster']).count().reset_index()
dataset.columns = ['ts','pickup_cluster','request_count']


# In[24]:


dataset.head()


# In[25]:


dataset.shape


# There should be around 878400 records if we take interval of 1 year in range on 30 mins of all the 366 days 

# In[27]:


from datetime import datetime, timedelta


# In[28]:


## Adding Dummy pickup cluster -1



# Create list of timestamps
l = [datetime(2020, 3, 26, 0, 0, 0) + timedelta(minutes=30*i) for i in range(0, 48*365)]

# Create list of lists for appending to DataFrame
lt = [[x, -1, 0] for x in l]

# Create temporary DataFrame
temp = pd.DataFrame(lt, columns=['ts', 'pickup_cluster', 'request_count'])

# Concatenate original dataset with temp DataFrame
dataset = pd.concat([dataset, temp], ignore_index=True)


# In[29]:


data = dataset.set_index(['ts', 'pickup_cluster']).unstack().fillna(value=0).asfreq(freq='30Min').stack().sort_index(level=1).reset_index()


# In[30]:


# Removing Dummy Cluster
data = data[data.pickup_cluster>=0]


# In[32]:


data.shape


# In[31]:


assert len(data)==878400


# Adding TimeFeatures

# In[33]:


data['mins'] = data.ts.dt.minute
data['hour'] = data.ts.dt.hour
data['day'] = data.ts.dt.day
data['month'] = data.ts.dt.month
data['dayofweek'] = data.ts.dt.dayofweek
data['quarter'] = data.ts.dt.quarter


# In[34]:


data.head(10)


# AIM: To forecast demand for a given latitude-longitude

# Metric: RMSE, how close we are able to predict ride demand to true value

# In[35]:


import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from math import sqrt, ceil, floor
import matplotlib.pyplot as plt
from xgboost import plot_importance
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from joblib import dump, load
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


data.dtypes


# changing the datatype of request_count from float to integer

# In[37]:


data['request_count'] = pd.to_numeric(data['request_count'], downcast=  'integer')
data.ts = pd.to_datetime(data.ts)
data.head(10)


# Redefining the schema of the dataset data

# In[39]:


data = data[['ts', 'pickup_cluster', 'mins', 'hour', 'month', 'quarter', 'dayofweek', 'request_count']]


# In[44]:


data.shape


# In[40]:


data


# - Train : first 23 days of the month
# - Test: last 7 days of every month

# In[46]:


data_train = data[data.ts.dt.day <= 23]
data_test = data[data.ts.dt.day > 23]


# In[47]:


data_train.shape


# In[48]:


data_test.shape


# Spliting data set into training and testing 
# - x: Feature columns {independent columns}
# - y: Target column {dependent columns}

# In[49]:


x_train = data_train.iloc[:, 1:-1]
y_train = data_train.iloc[:,-1]


# In[50]:


x_test = data_test.iloc[:, 1:-1]
y_test = data_test.iloc[:,-1]


# Making a function to calculate a performance metric for a regression model 

# In[51]:


def metrics_calculate(regressor):
    y_pred = regressor.predict(x_test) #making prediction based on x_test
    rms = sqrt(mean_squared_error(y_test, y_pred)) #calculate rmse between y_test and y_pred
    return rms


# Testing 1
# - features as = pickup_cluster, mins, hour, month, quarter, dayofweek

# Random Forest Regression
# - n_estimators : no of trees
# - random_state : randomness of taking a particular sequence of number 
# - n_jobs : to use all available CPU cores
# - verbose : to provide more detail info about for process

# In[52]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state=42, n_jobs = -1, verbose=True)
regressor.fit(x_train,y_train)
print("RMSE TRAIN: {}, RMSE TEST:{}".format(sqrt(mean_squared_error(y_train, regressor.predict(x_train))), metrics_calculate(regressor)))


# In[53]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state=42, n_jobs = -1)
regressor.fit(x_train,y_train)
print("RMSE TRAIN: {}, RMSE TEST:{}".format(sqrt(mean_squared_error(y_train, regressor.predict(x_train))), metrics_calculate(regressor)))


# In[54]:


feature_importances = pd.DataFrame(regressor.feature_importances_,
                                   index = x_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances


# Form the observation it is seen that the Random Forest is somewhat overfitting
# -  rmse value for test is greater, that means it performs well in training data but not on test data

# Boosting Algorithm

# XGBoost
# - learning_rate : to prevent the model form ovefitting
# - random_state :  randomness of taking a particular sequence of number in a batch to iterate in the loop 
# - n_estimators : no of boosting trees
# - max_depth : depth of each tree
# - objective : to minimize the loss function during training
# 

# In[56]:


import xgboost as xgb
model=xgb.XGBRegressor(learning_rate=0.01, random_state=0, n_estimators=1000, max_depth=8, objective="reg:squarederror")

eval_set = [(x_test, y_test)] #to monitor models performance on unseen data
model.fit(x_train,y_train,verbose=True, eval_set=eval_set, early_stopping_rounds=15,eval_metric="rmse")
print("XGBOOST Regressor")
print("Model Score:",model.score(x_train,y_train))
print("RMSE TRAIN: {}, RMSE TEST:{}".format(sqrt(mean_squared_error(y_train, model.predict(x_train))), metrics_calculate(model)))


# In[59]:


data = {
    'Algorithm': ['Random Forest', 'XGBoost'],
    'RMSE Train': [1.9048, 2.5953],
    'RMSE Test': [4.3232, 4.3232]
}
Result = pd.DataFrame(data)


# In[60]:


Result


# **As the Random Forest Algorithm was overfitting and afftecting the accuracy of the model we went ahead with XGBoost algorithm thorugh that the accuracy came out to be 85.85%**
