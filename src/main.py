import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from imblearn.over_sampling import SMOTE

#saving model to path
def save_model(model, filepath):
    with open(filepath, "wb") as file:
        pickle.dump(model, file)

#loading model to path
def load_model(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)

#load data
def load_data(filepath):
    return pd.read_csv(filepath)

#normalizing the features to standardize our range from 0 to 1
def normalize_data(X_train, X_test):
    scaler = MinMaxScaler() # create a scaler object
    X_train_normalized = scaler.fit_transform(X_train) #fit_transform() calculates min and max value for each feature
    X_test_normalized = scaler.transform(X_test) #transform() rescales based on min and max values
    return X_train_normalized, X_test_normalized, scaler

#box-cox transformation for our data
def apply_boxcox(data):
    skewed_columns = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"] #applies box-cox to these features
    
    for col in skewed_columns: #goes through data and checks if its positive 
        if (data[col] <= 0).any():  #if data is negative 
            shift_value = abs(data[col].min()) + 1  #adds #1 to make it positive
            data[col] += shift_value #ensures that every value in the column becomes positive


    for col in skewed_columns: #applying boxcox to each column
        data[col], _ = stats.boxcox(data[col]) #stats.boxcox() implements the most optimal lambda for us
    
    return data

def oversampling(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote

#plotting heatmap
def plot_heatmap(data):
    plt.figure(figsize=(10, 8))  #set figure size
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")  #plot heatmap
    plt.title("Correlation Heatmap of Features")  
    plt.show()  

#split data into X and y
def split_data(data):
    X = data.drop(columns="Type")   #remove Type column and stores others in X
    y = data["Type"]     #store Type column in y
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)   #split into 80% training size and 20% test size


#creating KNN Model with K = 3
def train_knn(X_train, y_train):

    knn = KNeighborsClassifier(n_neighbors=3)  # Create the KNN model
    knn.fit(X_train, y_train)  # Train the KNN model
    return knn  # Return both the model and the scaler

#function to evaluate the model
def evaluate_model(knn, X_test, y_test):
    y_pred = knn.predict(X_test) #uses trained model to predict labels
    print("Confusion Matrix:") #print confusion matrix 
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:") #print detailed statistical report
    print(classification_report(y_test, y_pred, zero_division=0))

#hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    param_grid = {"n_neighbors": [3, 5, 7, 9, 11, 15, 21]} #excluding 1 and including only odd numbers to always get a result with majority voting
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=6) #cv=6 == cross validating 6 times
    grid_search.fit(X_train, y_train) #use X_train, y_train for grid_search
    return grid_search.best_estimator_.n_neighbors #returns best value for k

#plot histograms
def plot_histograms(data):
    data.drop(columns="Type").hist(bins=10, figsize=(10, 8)) #removes type column from data -> generates histogram with 20 bins and size of (10,8) inches
    plt.suptitle("Skewness calculation")
    plt.show()

 
#check if our data is skewed
def calculate_skewness(data):
    skewness = data.drop(columns="Type").skew()
    print("Skewness: ")
    print(skewness)

def predict_class(knn_model, entries, labels, scaler):
    try:
        #collect input data
        input_data = np.array([[float(entries[label].get()) for label in labels]])

        #turn into data frame to pass appropriate feature name
        input_data_df = pd.DataFrame(input_data, columns=labels)

        #normalize the input data using the same scaler as during training
        input_data_normalized = scaler.transform(input_data_df)
        

        #make prediction
        prediction = knn_model.predict(input_data_normalized)
        confidence_score = np.max(knn_model.predict_proba(input_data_normalized)) #probability prediction of each clsas given by input_data_normalized

        messagebox.showinfo("Prediction", f"Predicted Class: {prediction[0]}\nConfidence: {confidence_score:.2f}")
    except ValueError:
        messagebox.showerror("Please enter valid numerical values.")

#main
def main():
    model_path = r"C:\Users\FMJ Shawty\Documents\knn_model.pkl" #path to model

    #load dataset
    filepath = r"C:\Users\FMJ Shawty\Desktop\UNI\5. Semester\Machine Learning\ML Glass Identification\data\glass.csv"
    data = load_data(filepath)
    #print(data)
    
    #apply Box-Cox
    data = apply_boxcox(data)

    #split and oversampling and normalizing data
    X_train, X_test, y_train, y_test = split_data(data)
    print("Class distribution in y_train:")
    print(y_train.value_counts())
    print("Class distribution in y_test:")
    print(y_test.value_counts())
    #X_train_smote, y_train_smote = oversampling(X_train, y_train)
    X_train_normalized, X_test_normalized, scaler = normalize_data(X_train, X_test)



    #train and evaluate KNN model
    knn = train_knn(X_train_normalized, y_train)
    evaluate_model(knn, X_test_normalized, y_test)

    save_model(knn, model_path)
    #print("Class distribution after SMOTE:")
    #print(pd.Series(y_train_smote).value_counts())

    knn_model = load_model(model_path)  #loading trained model

    root = tk.Tk()  #init GUI window
    root.title("Glass Identification")
    #GUI setup
    labels = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"] #defining labels
    entries = {} #stores text fields for user input

    for label in labels: #create input fields for each label to classify glass type
        tk.Label(root, text=label).grid(row=labels.index(label), column=0) #text label for each feature
        entry = tk.Entry(root) #input field for user entry
        entry.grid(row=labels.index(label), column=1) #aligning positioning
        entries[label] = entry #stores entry widget in entries dictionary using label as key


    #create button to cast predict of glass types
    predict_button = tk.Button(root, text="Predict", command=lambda: predict_class(knn, entries, labels, scaler)) #button labeled "predict" command=lambda creates function to pass arg
    predict_button.grid(row=len(labels), column=0, columnspan=2) #button positioning

    def show_histogram():
        plot_histograms(data)
        
    def show_heatmap():
        plot_heatmap(data)

    #create buttons to show heatmap and histogram
    heatmap_button = tk.Button(root, text="Show Heatmap", command=show_heatmap)
    heatmap_button.grid(row=len(labels) + 1, column=0, columnspan=2)

    histogram_button = tk.Button(root, text="Show Histogram", command=show_histogram)
    histogram_button.grid(row=len(labels) + 2, column=0, columnspan=2)
    root.mainloop()

    #hyperparameter tuning
    #best_k = tune_hyperparameters(X_train_normalized, y_train)
    #print("Best k :", best_k)


    #check transformed skewness
    #print("Transformed Skewness:")
    #print(transformed_data.skew())

main()