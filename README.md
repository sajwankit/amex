# American Express - Default Prediction 
#### Predict if a customer will default in the future

**Overview:**
https://www.kaggle.com/competitions/amex-default-prediction

**Objective:**
> The objective is to predict the probability that a customer does not pay back their credit card balance amount in the future based on their monthly customer profile.

**Data:**  
The dataset contains aggregated profile features for each customer at each statement date. Features are anonymized and normalized, and fall into the following general categories:  
  
D_* = Delinquency variables  
S_* = Spend variables  
P_* = Payment variables  
B_* = Balance variables  
R_* = Risk variables  
with the following features being categorical:  
  
['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']  
  
**Our task is to predict, for each customer_ID, the probability of a future payment default (target = 1).**  
  
Files
- train_data.csv - training data with multiple statement dates per customer_ID
- train_labels.csv - target label for each customer_ID
- test_data.csv - corresponding test data; your objective is to predict the target label for each customer_ID
- sample_submission.csv - a sample submission file in the correct format

APPROACH
We build and train an XGBoost model using competition data . Then we engineer features suggested by @huseyincot in his notebooks here and here. This XGB model achieves CV 0.792 LB 0.793! When training with XGB, we use a special XGB dataloader called DeviceQuantileDMatrix which uses a small GPU memory footprint. This allows us to engineer more additional columns and train with more rows of data. Our feature engineering is performed using RAPIDS on the GPU to create new features quickly.
