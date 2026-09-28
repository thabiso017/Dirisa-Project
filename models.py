import pandas as pd
import numpy as np

# 1. Load Raw Election Data
gauteng = pd.read_csv("GP.csv")
kzn = pd.read_csv("KN.csv")

# Combine both provinces
raw = pd.concat([gauteng, kzn], ignore_index=True)

# Keep PR ballot (party vote)
pr = raw[raw['BallotType'] == 'PR'].copy()

# Clean WardID
pr['WardID'] = pr['Ward'].str.replace('Ward ', '', regex=False).str.strip()

# Aggregate to Station Level
station_level = pr.groupby(['Province', 'Municipality', 'WardID', 'VotingDistrict'], as_index=False).agg(
    Registered_Voters=('RegisteredVoters', 'first'),
    Spoilt_Votes=('SpoiltVotes', 'first'),
    Total_Votes_Cast=('TotalValidVotes', 'sum')
)

# Aggregate to Ward Level
ward_level = station_level.groupby(['Province', 'Municipality', 'WardID'], as_index=False).agg(
    Registered_Voters=('Registered_Voters', 'sum'),
    Total_Votes_Cast=('Total_Votes_Cast', 'sum'),
    Spoilt_Votes=('Spoilt_Votes', 'sum')
)

# Calculate Turnout Percentage
ward_level['Turnout_Pct'] = (
    (ward_level['Total_Votes_Cast'] + ward_level['Spoilt_Votes']) / ward_level['Registered_Voters'] * 100
).round(2)

# Save LGE 2021 Results
final = ward_level[['WardID', 'Registered_Voters', 'Total_Votes_Cast', 'Spoilt_Votes', 'Turnout_Pct']]
final.to_csv('lge_2021_results.csv', index=False)

# 2. Load Demographic Registration Data
registration = pd.read_csv('iec_voter_registration_progress.csv')

registration_gk = registration[
    registration['Province'].isin(['Gauteng', 'KwaZulu-Natal'])
].copy()

registration_gk['Municipality'] = registration_gk['Municipality'].replace({
    'KZN275 - Inkosi uMtubatuba': 'KZN275 - Mtubatuba',
    'KZN434 - Johannes Phumani Pungula': 'KZN434 - uBuhlebezwe'
})

# Merge Datasets
model_data = ward_level.merge(
    registration_gk[
        [
            'Province',
            'Municipality',
            'Total_Registered',
            'Youth_18_34_Total',
            'Female_Percentage',
            'Total_Male',
            'Total_Female'
        ]
    ],
    on=['Province', 'Municipality'],
    how='left'
)

model_data['Youth_Registration_Rate'] = (
    model_data['Youth_18_34_Total'] / model_data['Total_Registered'] * 100
).round(2)

model_data['Female_Male_Ratio'] = (
    model_data['Total_Female'] / model_data['Total_Male']
).round(3)

# 3. Model Training
X = model_data[['Youth_Registration_Rate', 'Female_Percentage', 'Female_Male_Ratio']]
y = model_data['Turnout_Pct']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# XGBoost
from xgboost import XGBRegressor
xgb_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

model_results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'MAE': [
        mean_absolute_error(y_test, y_pred_lr),
        mean_absolute_error(y_test, y_pred_rf),
        mean_absolute_error(y_test, y_pred_xgb)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    ],
    'R2': [
        r2_score(y_test, y_pred_lr),
        r2_score(y_test, y_pred_rf),
        r2_score(y_test, y_pred_xgb)
    ]
}).round(2)

print("Models Trained Successfully:")
print(model_results)