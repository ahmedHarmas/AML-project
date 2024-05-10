#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Define global variables
FILE_PATH = "Raisin_Dataset1.csv"
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
model = None  # Initialize the model as None

def main():
    global model  # Use the global model variable
    # Load the dataset
    dataset = pd.read_csv(FILE_PATH)

    # Split into features and target
    X = dataset.drop('Class', axis=1).astype(float)
    X = X.drop(['Area', 'ConvexArea'], axis=1)  # Dropping 'Area' and 'ConvexArea' columns
    y = dataset['Class']

    # Normalize the features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Split into training, validation, and test sets
    X_train, X_val_test, y_train, y_val_test = train_test_split(X_normalized, y, test_size=TEST_SIZE + VALIDATION_SIZE, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE), random_state=RANDOM_STATE)

    # Define the input shape
    input_shape = X_train.shape[1]

    # Define and compile the model
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(16, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # Train the model with validation
    history = model.fit(X_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(X_val, y_val))

    # Plot accuracy and loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.2f}")

    # Calculate and plot precision, recall, and F1 score
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")

    scores = {'Precision': precision, 'Recall': recall, 'F1 score': f1}
    df = pd.DataFrame(scores, index=['Test'])
    df.plot(kind='bar')
    plt.title('Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.show()

    # Print the classification report
    print(classification_report(y_test, y_pred))

def predict():
    if model is None:
        messagebox.showwarning("Model Not Trained", "Please train the model first.")
        return

    major_axis_length = float(entry_fields["MajorAxisLength"].get())
    minor_axis_length = float(entry_fields["MinorAxisLength"].get())
    eccentricity = float(entry_fields["Eccentricity"].get())
    extent = float(entry_fields["Extent"].get())
    perimeter = float(entry_fields["Perimeter"].get())

    user_input_array = np.array(
        [major_axis_length, minor_axis_length, eccentricity, extent, perimeter])

    # Use the global model for prediction
    prediction = model.predict(user_input_array.reshape(1, -1))  # Reshape for single prediction
    prediction_label = "Kecimen" if prediction >= 0.5 else "Besni"
    messagebox.showinfo("Prediction", f"The predicted class is: {prediction_label}")

def create_gui():
    root = tk.Tk()
    root.title("Model Trainer")

    # Labels and Entry Fields for user input
    fields = ["MajorAxisLength", "MinorAxisLength", "Eccentricity", "Extent", "Perimeter"]
    global entry_fields
    entry_fields = {}
    defaults = [585.9819941, 281.6014086, 0.876960072, 0.750683811, 1499.355]  # Default values
    for field, default in zip(fields, defaults):
        row = tk.Frame(root)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        label = tk.Label(row, width=15, text=field, anchor='w')
        entry = tk.Entry(row)
        entry.insert(0, str(default))  # Set default value
        label.pack(side=tk.LEFT)
        entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entry_fields[field] = entry

    # Run Button
    run_button = tk.Button(root, text="Run Model", command=main)
    run_button.pack()

    # Predict Button
    predict_button = tk.Button(root, text="Predict", command=predict)
    predict_button.pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui()


# In[ ]:




