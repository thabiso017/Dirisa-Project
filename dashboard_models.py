from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

DATA_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS = [
    "Youth_Registration_Rate",
    "Female_Percentage",
    "Female_Male_Ratio",
]

raw = pd.concat(
    [
        pd.read_csv(DATA_DIR / "GP.csv"),
        pd.read_csv(DATA_DIR / "KN.csv"),
    ],
    ignore_index=True,
)
pr = raw.loc[raw["BallotType"].eq("PR")].copy()
pr["WardID"] = (
    pr["Ward"].astype("string").str.replace("Ward ", "", regex=False).str.strip()
)

station_level = pr.groupby(
    ["Province", "Municipality", "WardID", "VotingDistrict"],
    as_index=False,
).agg(
    Registered_Voters=("RegisteredVoters", "first"),
    Spoilt_Votes=("SpoiltVotes", "first"),
    Total_Votes_Cast=("TotalValidVotes", "sum"),
)
ward_level = station_level.groupby(
    ["Province", "Municipality", "WardID"], as_index=False
).agg(
    Registered_Voters=("Registered_Voters", "sum"),
    Total_Votes_Cast=("Total_Votes_Cast", "sum"),
    Spoilt_Votes=("Spoilt_Votes", "sum"),
)
ward_level["Turnout_Pct"] = (
    (ward_level["Total_Votes_Cast"] + ward_level["Spoilt_Votes"])
    / ward_level["Registered_Voters"]
    * 100
).round(2)

registration = pd.read_csv(DATA_DIR / "iec_voter_registration_progress.csv")
registration = registration.loc[
    registration["Province"].isin(["Gauteng", "KwaZulu-Natal"])
].copy()
registration["Municipality"] = registration["Municipality"].replace(
    {
        "KZN275 - Inkosi uMtubatuba": "KZN275 - Mtubatuba",
        "KZN434 - Johannes Phumani Pungula": "KZN434 - uBuhlebezwe",
    }
)

model_data = ward_level.merge(
    registration[
        [
            "Province",
            "Municipality",
            "Total_Registered",
            "Youth_18_34_Total",
            "Female_Percentage",
            "Total_Male",
            "Total_Female",
        ]
    ],
    on=["Province", "Municipality"],
    how="left",
)
model_data["Youth_Registration_Rate"] = (
    model_data["Youth_18_34_Total"] / model_data["Total_Registered"] * 100
).round(2)
model_data["Female_Male_Ratio"] = (
    model_data["Total_Female"] / model_data["Total_Male"]
).round(3)
model_data = model_data.dropna(subset=FEATURE_COLUMNS + ["Turnout_Pct"]).copy()

if model_data.empty:
    raise ValueError(
        "No wards have complete turnout and registration data. Check the "
        "municipality names in the CSV files."
    )

X = model_data[FEATURE_COLUMNS]
y = model_data["Turnout_Pct"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression().fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(
    X_train, y_train
)
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
).fit(X_train, y_train)

predictions = {
    "Linear Regression": model.predict(X_test),
    "Random Forest": rf_model.predict(X_test),
    "XGBoost": xgb_model.predict(X_test),
}
model_results = pd.DataFrame(
    [
        {
            "Model": name,
            "MAE": mean_absolute_error(y_test, prediction),
            "RMSE": np.sqrt(mean_squared_error(y_test, prediction)),
            "R2": r2_score(y_test, prediction),
        }
        for name, prediction in predictions.items()
    ]
).round(2)
