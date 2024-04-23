from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

app = Flask("prediction")

# Load the dataset
matches = pd.read_csv("data.csv")

# Sample date conversion function for the specific format "YYYY年MM月DD日"
def convert_date(date_str):
    if '年' in date_str and '月' in date_str and '日' in date_str:
        parts = date_str.split('年')
        year = parts[0]
        month_day = parts[1].split('月')
        month = month_day[0]
        day = month_day[1].rstrip('日')
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    else:
        return date_str

matches["date"] = matches["date"].apply(convert_date)
matches["date"] = pd.to_datetime(matches["date"], errors='coerce')

# Convert 'result' to categorical class (W: 0, L: 1, D: 2)
matches["target"] = matches["result"].map({'W': 0, 'L': 1, 'D': 2}).astype("int")
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek

additional_predictors = ["xg", "xga", "poss", "attendance", "sh", "sot", "dist", "pk", "fk", "pkatt"]
predictors = ["venue_code", "opp_code", "hour", "day_code"] + additional_predictors

train, test = train_test_split(matches, test_size=0.3, random_state=42)

train_target = to_categorical(train['target'], num_classes=3)
test_target = to_categorical(test['target'], num_classes=3)

imputer = SimpleImputer(strategy="mean")
train_imputed = pd.DataFrame(imputer.fit_transform(train[predictors]), columns=predictors)
test_imputed = pd.DataFrame(imputer.transform(test[predictors]), columns=predictors)

scaler = StandardScaler()
train_imputed_scaled = scaler.fit_transform(train_imputed)
test_imputed_scaled = scaler.transform(test_imputed)

# Model definition
model = Sequential([
    Dense(128, activation="relu", kernel_regularizer=l2(0.01), input_shape=(len(predictors),)),
    Dropout(0.5),
    Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(3, activation="softmax")  # For 3 output classes
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    train_imputed_scaled,
    train_target,
    epochs=100,
    batch_size=32,
    validation_data=(test_imputed_scaled, test_target),
    verbose=0
)

def make_predictions_nn(data, team, predictors, imputer, scaler, model):
    team_matches = data[data["team"] == team]
    if team_matches.empty:
        return None  # Indicates no data for the team
    
    team_data = pd.DataFrame(imputer.transform(team_matches[predictors]), columns=predictors)
    team_imputed_scaled = scaler.transform(team_data)
    preds = model.predict(team_imputed_scaled)
    pred_results = [np.argmax(pred) for pred in preds]  # Get index of highest probability
    result_map = {0: 'W', 1: 'L', 2: 'D'}  # Map index to result
    pred_results = [result_map[pred] for pred in pred_results]

    results = []
    correct_predictions = 0
    for match, pred_result in zip(team_matches.itertuples(index=False), pred_results):
        match_info = {
            "team": team,
            "opponent": match.opponent,
            "prediction": pred_result,
            "actual_result": match.result,
            "venue": match.venue,
            "date": match.date
        }
        results.append(match_info)
        if pred_result == match.result:
            correct_predictions += 1
    accuracy = correct_predictions / len(pred_results) * 100 if pred_results else 0
    
    output_data = {
        "accuracy": accuracy,
        "results": results,
    }
    return output_data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    team = data.get('team')
    if not team:
        return jsonify({"error": "Team name is required"}), 400

    prediction_result = make_predictions_nn(matches, team, predictors, imputer, scaler, model)
    if prediction_result is None:
        return jsonify({"error": "Team not found"}), 404

    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
