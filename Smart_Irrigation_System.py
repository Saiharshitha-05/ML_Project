import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json
import os
from datetime import datetime

# Load and prepare data
data = pd.read_csv("C:/Users/DELL/Desktop/MLproject/sensor_data6.csv")
label_encoder = LabelEncoder()
data['crop_type'] = label_encoder.fit_transform(data['crop_type'])

features = ['temperature', 'humidity', 'soil_moisture', 'rainfall', 'wind_speed', 'crop_type']
X = data[features]
y = data['irrigation']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# GUI setup
root = tk.Tk()
root.title("Irrigation Prediction")

entries = {}
fields = ['Temperature', 'Humidity', 'Soil Moisture', 'Rainfall', 'Wind Speed']

for idx, field in enumerate(fields):
    label = tk.Label(root, text=field)
    label.grid(row=idx, column=0)
    entry = tk.Entry(root)
    entry.grid(row=idx, column=1)
    entries[field] = entry

# Crop type dropdown
tk.Label(root, text="Crop Type").grid(row=len(fields), column=0)
crop_var = tk.StringVar()
crop_dropdown = tk.OptionMenu(root, crop_var, *label_encoder.classes_)
crop_dropdown.grid(row=len(fields), column=1)
crop_var.set(label_encoder.classes_[0])

# Prediction function with JSON logging
def predict():
    try:
        # Get user input
        input_data = [float(entries[f].get()) for f in fields]
        crop = crop_var.get()
        crop_encoded = label_encoder.transform([crop])[0]
        input_data.append(crop_encoded)

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data], columns=features)

        # Prediction and confidence
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = round(probabilities[prediction] * 100, 2)

        # Result message
        result = "Irrigation Needed" if prediction == 1 else "No Irrigation Needed"
        messagebox.showinfo("Prediction", f"{result}\nConfidence: {confidence}%")

        # Prepare log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "temperature": input_data[0],
            "humidity": input_data[1],
            "soil_moisture": input_data[2],
            "rainfall": input_data[3],
            "wind_speed": input_data[4],
            "crop_type": crop,
            "prediction_result": result,
            "confidence_percent": confidence
        }

        # Save to JSON file on Desktop
        json_file = "C:/Users/DELL/Desktop/MLproject/predictions_log.json"
        print("Saving to:", json_file)

        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                json_data = json.load(f)
        else:
            json_data = []

        json_data.append(log_entry)

        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=4)

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {str(e)}")

# Predict button
btn = tk.Button(root, text="Predict", command=predict)
btn.grid(row=len(fields)+1, columnspan=2, pady=10)

root.mainloop()
