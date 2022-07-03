# American Express - Default Prediction 
### Predict if a customer will default in the future

#### OVERVIEW
https://www.kaggle.com/competitions/amex-default-prediction
  
  
#### OBJECTIVE
> The objective is to predict the probability that a customer does not pay back their credit card balance amount in the future based on their monthly customer profile.  
  
  
#### DATA
The dataset contains aggregated profile features for each customer at each statement date. Features are anonymized and normalized, and fall into the following general categories:  
  
D_* = Delinquency variables  
S_* = Spend variables  
P_* = Payment variables  
B_* = Balance variables  
R_* = Risk variables  
with the following features being categorical:  
  
['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']  
  
> **The task is to predict, for each customer_ID, the probability of a future payment default (target = 1).**  
 
#### APPROACH (WIP)
We build and train an **XGBoost model** using customer data, this XGB model achieves CV score of 0.792.

> When training with XGB, we use a special XGB dataloader called **[DeviceQuantileDMatrix](https://xgboost.readthedocs.io/en/latest/python/examples/quantile_data_iterator.html "DeviceQuantileDMatrix")** which uses a small GPU memory footprint. This allows us to engineer more additional columns and train with more rows of data.

Our feature engineering is performed using **[RAPIDS](https://rapids.ai/ "RAPIDS")** on the GPU to create new features quickly.  

#### TRAINING LOGS
We use a centralised logging mechanism for whole project using **python logging module**. One of the xgboost model training for 3 folds of data can be found at:  
**https://github.com/sajwankit/amex/logs/train_4July2022.log**
