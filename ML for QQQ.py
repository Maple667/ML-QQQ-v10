import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import os
import json
from tkinter import simpledialog
from tkinter import filedialog
from tkinter import messagebox
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# Load the data
data = pd.read_csv('QQQdata.csv')
data['Date'] = pd.to_datetime(data['Date'])  # Convert 'date' column to datetime format

# Specify the feature columns
features = ['Open', 'High', 'Low', 'Close', 'Close1', 'Close2', 'Fib', 'Fib 0.236', 'Fib 0.382', 'Fib 0.5', 'Fib 0.618', 
            'Fib 0.764', 'Fib 1', 'Fib -0.236', 'Fib -0.382', 'Fib -0.5', 'Fib -0.618', 'Fib -0.764', 'Fib -1', 
            'VIX', 'VIX1', 'VIX2', 'VIX3', 'Plot', 'Trend', 'Trend2', '%R', '%R1', '%R2', '%R3', 'EMA', 'EMA1', 
            'EMA2', 'EMA3', 'RSI', 'RSI1', 'RSI2', 'RSI3', 'RSI MA', 'RSI MA1', 'RSI MA2', 'RSI MA3', 'TCI', 'TCI1', 
            'TCI2', 'TCI3', 'wt2', 'wt21', 'wt22', 'wt23', 'wtdiff', 'wtdiff1', 'wtdiff2', 'wtdiff3', 'Val', 'Val1', 
            'Val2', 'Val3', 'Val4', 'Squeeze', 'Vix Fix', 'Vix Fix1', 'Vix Fix2', 'Vix Fix3', 'DI+', 'DI+1', 'DI+2', 
            'DI+3', 'DI-', 'DI-1', 'DI-2', 'DI-3', 'ADX', 'ADX1', 'ADX2', 'ADX3']

def save_settings():
    settings = {f: var.get() for f, var in feature_vars.items()}
    settings['output_column'] = output_column.get()  # Save the output text
    settings_filename = filedialog.asksaveasfilename(initialdir="Config", defaultextension=".json")
    with open(settings_filename, 'w') as file:
        json.dump(settings, file)

def import_settings():
    settings_filename = filedialog.askopenfilename(initialdir="Config", defaultextension=".json")
    with open(settings_filename, 'r') as file:
        settings = json.load(file)
    for feature, var in feature_vars.items():
        var.set(settings.get(feature, 0))  # Use the settings value if it exists, otherwise default to 0
    output_column.set(settings.get('output_column', ''))  # Load the output text, default to empty string if not found


# Define function to perform the data processing and prediction
from sklearn.preprocessing import StandardScaler

# Prepare data for ML
def get_data():

    try:
        target = [output_column.get()]
        if not target[0]:  
            messagebox.showerror("Error", "Output box is empty. Please fill it before starting.")
            return None, None, None, None, None, None

        train_start_date = train_start_date_entry.get_date()
        train_end_date = train_end_date_entry.get_date()
        predict_start_date = predict_start_date_entry.get_date()
        predict_end_date = predict_end_date_entry.get_date()

        train_start_date = train_start_date.strftime("%Y-%m-%d")
        train_end_date = train_end_date.strftime("%Y-%m-%d")
        predict_start_date = predict_start_date.strftime("%Y-%m-%d")
        predict_end_date = predict_end_date.strftime("%Y-%m-%d")

        train_data = data[(data['Date'] >= train_start_date) & (data['Date'] <= train_end_date)]
        predict_data = data[(data['Date'] >= predict_start_date) & (data['Date'] <= predict_end_date)]

        selected_features = [feature for feature, var in feature_vars.items() if var.get()]
        X = train_data[selected_features]
        y = train_data[target]

        data_majority = train_data[train_data.outcome==0]
        data_minority1 = train_data[train_data.outcome==-1]
        data_minority2 = train_data[train_data.outcome==1]

        data_minority1_upsampled = resample(data_minority1, replace=True, n_samples=data_majority.shape[0], random_state=42)
        data_minority2_upsampled = resample(data_minority2, replace=True, n_samples=data_majority.shape[0], random_state=42)

        data_upsampled = pd.concat([data_majority, data_minority1_upsampled, data_minority2_upsampled])

        X_train, X_test, y_train, y_test = train_test_split(data_upsampled[selected_features], data_upsampled[target], test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, predict_data, selected_features, scaler

    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None, None, None, None, None, None
    
# Perform RandomForestClassification
def RandomForestClassification(X_train_scaled, X_test_scaled, y_train, y_test, predict_data, selected_features, scaler):
    rf = RandomForestClassifier(n_estimators=128, bootstrap=True, random_state=42)
    rf.fit(X_train_scaled, y_train.values.ravel())

    y_test_pred = rf.predict(X_test_scaled)
    accuracy = rf.score(X_test_scaled, y_test)
    print(f"Model accuracy: {accuracy}")

    new_data = predict_data
    X_new = new_data[selected_features]
    X_new_scaled = scaler.transform(X_new)

    new_predictions = rf.predict(X_new_scaled)
    return pd.DataFrame({
        'Date': new_data['Date'],
        'Time': new_data['Time'],
        'Random Forest': new_predictions
    })

# Perform Decision Tree
def DecisionTree(X_train_scaled, X_test_scaled, y_train, y_test, predict_data, selected_features, scaler):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_scaled, y_train.values.ravel())

    y_test_pred = dt.predict(X_test_scaled)
    accuracy = dt.score(X_test_scaled, y_test)
    print(f"Model accuracy: {accuracy}")

    new_data = predict_data
    X_new = new_data[selected_features]
    X_new_scaled = scaler.transform(X_new)

    new_predictions = dt.predict(X_new_scaled)
    return pd.DataFrame({
        'Date': new_data['Date'],
        'Time': new_data['Time'],
        'Decision Tree': new_predictions
    })

from sklearn.neighbors import KNeighborsClassifier

def KNN(X_train_scaled, X_test_scaled, y_train, y_test, predict_data, selected_features, scaler):
    knn = KNeighborsClassifier(n_neighbors=5) # You can adjust the number of neighbors
    knn.fit(X_train_scaled, y_train.values.ravel())

    y_test_pred = knn.predict(X_test_scaled)
    accuracy = knn.score(X_test_scaled, y_test)
    print(f"Model accuracy: {accuracy}")

    new_data = predict_data
    X_new = new_data[selected_features]
    X_new_scaled = scaler.transform(X_new)

    new_predictions = knn.predict(X_new_scaled)
    return pd.DataFrame({
        'Date': new_data['Date'],
        'Time': new_data['Time'],
        'KNN': new_predictions
    })

def SVM(X_train_scaled, X_test_scaled, y_train, y_test, predict_data, selected_features, scaler):
    svm = SVC(kernel='rbf', random_state=42) # You can adjust the kernel to linear, poly or sigmoid
    svm.fit(X_train_scaled, y_train.values.ravel())

    y_test_pred = svm.predict(X_test_scaled)
    accuracy = svm.score(X_test_scaled, y_test)
    print(f"Model accuracy: {accuracy}")

    new_data = predict_data
    X_new = new_data[selected_features]
    X_new_scaled = scaler.transform(X_new)

    new_predictions = SVM.predict(X_new_scaled)
    return pd.DataFrame({
        'Date': new_data['Date'],
        'Time': new_data['Time'],
        'SVM': new_predictions
    })

def train_model():
    model_func_mapping = {'RandomForest': RandomForestClassification,
                          'DecisionTree': DecisionTree,
                          'KNN': KNN,
                          'SVM': SVM}

    if X_train_scaled is not None:
        # Initialize an empty list to hold the prediction dataframes
        predictions_dataframes = []

        # Loop through all models and call the function if the model is selected
        for model_name, func in model_func_mapping.items():
            if ml_model_vars[model_name].get() == 1:
                df = func(X_train_scaled, X_test_scaled, y_train, y_test, predict_data, selected_features, scaler)
                predictions_dataframes.append(df)

        # Export all predictions
        Export(predictions_dataframes)

def Export(predictions_dataframes):
    # Join all prediction dataframes on 'Date' and 'Time'
    final_df = pd.concat(predictions_dataframes, axis=1)
    final_df = final_df.loc[:,~final_df.columns.duplicated()]

    base_filename = 'new_data_with_predictions'
    file_counter = 0
    filename = f'{base_filename}.csv'
    while True:
        try:
            with open(filename, 'a') as file:
                pass
            break
        except IOError:
            file_counter += 1
            filename = f'{base_filename}_{file_counter}.csv'

    try:
        final_df.to_csv(filename, index=False)
        messagebox.showinfo("Success", "Data exported successfully to " + filename)
    except Exception as e:
        messagebox.showerror("Error", "Could not export data to " + filename + "\n\n" + str(e))

def select_all():
    for var in feature_vars.values():
        var.set(1)

def unselect_all():
    for var in feature_vars.values():
        var.set(0)

root = tk.Tk()

# Set default dates
train_start_date = datetime.strptime('01/01/2020', '%m/%d/%Y')
train_end_date = datetime.strptime('12/31/2022', '%m/%d/%Y')
predict_start_date = datetime.strptime('01/01/2023', '%m/%d/%Y')
predict_end_date = datetime.strptime('07/18/2023', '%m/%d/%Y')

# Training date range
tk.Label(root, text="Training Start Date").grid(row=0, column=0)
train_start_date_entry = DateEntry(root)
train_start_date_entry.set_date(train_start_date)
train_start_date_entry.grid(row=0, column=1)

tk.Label(root, text="Training End Date").grid(row=1, column=0)
train_end_date_entry = DateEntry(root)
train_end_date_entry.set_date(train_end_date)
train_end_date_entry.grid(row=1, column=1)

# Prediction date range
tk.Label(root, text="Prediction Start Date").grid(row=2, column=0)
predict_start_date_entry = DateEntry(root)
predict_start_date_entry.set_date(predict_start_date)
predict_start_date_entry.grid(row=2, column=1)

tk.Label(root, text="Prediction End Date").grid(row=3, column=0)
predict_end_date_entry = DateEntry(root)
predict_end_date_entry.set_date(predict_end_date)
predict_end_date_entry.grid(row=3, column=1)

# Calculate the last row used by the checkboxes
last_checkbox_row = len(features) // 3 + 5

# Button row will be one more than the last checkbox row
button_row = last_checkbox_row + 3

# Ask user for what the Output Column is called
tk.Label(root, text="Output").grid(row=button_row, column=0)
output_column = tk.StringVar()
output_column_entry = tk.Entry(root, textvariable=output_column)
output_column_entry.grid(row=button_row, column=1)

# Variable to hold the selected option
model_selection = tk.StringVar()

ml_model_vars = {"Random Forest": tk.IntVar(),
                 "Decision Tree": tk.IntVar(),
                 "KNN": tk.IntVar(),
                 "SVM": tk.IntVar()}

ml_model_frame = tk.LabelFrame(root, text="ML Model Selection")
ml_model_frame.grid(row=4, column=0, columnspan=2, sticky='w')

for i, (model_name, var) in enumerate(ml_model_vars.items()):
    cb = ttk.Checkbutton(ml_model_frame, text=model_name, variable=var)
    cb.grid(row=i // 2, column=i % 2, sticky='w')

# Feature selection default values
feature_vars = {feature: tk.IntVar(value=1) for feature in features}
feature_frame = tk.LabelFrame(root, text="Feature Selection")
feature_frame.grid(row=5, column=0, columnspan=3)
for i, (feature, var) in enumerate(feature_vars.items(), start=5):
    cb = ttk.Checkbutton(feature_frame, text=feature, variable=var)
    cb.grid(row=i // 3, column=i % 3, sticky='w')

# Button to select all checkboxes
select_all_button = tk.Button(feature_frame, text="Select All", command=select_all)
select_all_button.grid(row=i // 3 + 1, column=0)

# Button to unselect all checkboxes
unselect_all_button = tk.Button(feature_frame, text="Unselect All", command=unselect_all)
unselect_all_button.grid(row=i // 3 + 1, column=1)

# Button to load data
load_data_button = tk.Button(feature_frame, text="Load Data", command=get_data)
load_data_button.grid(row=i // 3 + 2, column=0, columnspan=2)

# Button to save settings
tk.Button(root, text="Save Settings", command=save_settings).grid(row=button_row+2, column=0)
tk.Button(root, text="Import Settings", command=import_settings).grid(row=button_row+2, column=1)

# Button to close the GUI
tk.Button(root, text="Close", command=root.destroy).grid(row=button_row+3, column=1)

# Button to start training
tk.Button(root, text="Train", command=train_model).grid(row=button_row+3, column=0)


root.mainloop()