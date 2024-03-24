from flask import Flask, render_template, request , url_for
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.impute import SimpleImputer
from werkzeug.datastructures import ImmutableMultiDict
from sklearn.metrics import accuracy_score
import warnings

app = Flask(__name__)

df = pd.read_csv('infantset.csv')  
def train_neural_network():
    df["BirthWeight"] = df["BirthWeight"].map({'WeightTooLow':3 ,'LowWeight':2,'NormalWeight':1})
    df["FamilyHistory"] = df["FamilyHistory"].map({'AboveTwoCases':3 ,'ZeroToTwoCases':2,'NoCases':1})
    df["PretermBirth"] = df["PretermBirth"].map({'4orMoreWeeksEarlier':3 ,'2To4weeksEarlier':2,'NotaPreTerm':1})
    df["HeartRate"] = df["HeartRate"].map({'RapidHeartRate':3 ,'HighHeartRate':2,'NormalHeartRate':1})
    df["BreathingDifficulty"] = df["BreathingDifficulty"].map({'HighBreathingDifficulty':3 ,'BreathingDifficulty':2,'NoBreathingDifficulty':1})
    df["SkinTinge"] = df["SkinTinge"].map({'Bluish':3 ,'LightBluish':2,'NotBluish':1})
    df["Responsiveness"] = df["Responsiveness"].map({'UnResponsive':3 ,'SemiResponsive':2,'Responsive':1})
    df["Movement"] = df["Movement"].map({'Diminished':3 ,'Decreased':2,'NormalMovement':1})
    df["DeliveryType"] = df["DeliveryType"].map({'C_Section':3 ,'DifficultDelivery':2,'NormalDelivery':1})
    df["MothersBPHistory"] = df["MothersBPHistory"].map({'VeryHighBP':3 ,'HighBP':2,'BPInRange':1})
    df["CardiacArrestChance"] = df["CardiacArrestChance"].map({'High':2 ,'Medium':1,'Low':0})
    data = df[["BirthWeight","FamilyHistory","PretermBirth","HeartRate","BreathingDifficulty","SkinTinge","Responsiveness","Movement","DeliveryType","MothersBPHistory","CardiacArrestChance"]].to_numpy()
    inputs = data[:,:-1]
    outputs = data[:, -1]
    warnings.filterwarnings("ignore")
    training_data = inputs[:1000]
    training_labels = outputs[:1000]
    test_data = inputs[1000:]
    test_labels = outputs[1000:]
    tf.keras.backend.clear_session()
    nn_model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.elu), 
                                    tf.keras.layers.Dense(64, activation=tf.nn.selu), 
                                    tf.keras.layers.Dense(32, activation=tf.nn.softmax), 
                                    tf.keras.layers.Dense(16, activation=tf.nn.softplus)])
    nn_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    nn_model.fit(training_data, training_labels, epochs=100)
    _, nn_accuracy = nn_model.evaluate(test_data, test_labels)
    return nn_model , nn_accuracy

def train_bagging_classifier():
    df1 = pd.read_csv('infantset.csv')
    df1["BirthWeight"] = df1["BirthWeight"].map({'WeightTooLow':3 ,'LowWeight':2,'NormalWeight':1})
    df1["FamilyHistory"] = df1["FamilyHistory"].map({'AboveTwoCases':3 ,'ZeroToTwoCases':2,'NoCases':1})
    df1["PretermBirth"] = df1["PretermBirth"].map({'4orMoreWeeksEarlier':3 ,'2To4weeksEarlier':2,'NotaPreTerm':1})
    df1["HeartRate"] = df1["HeartRate"].map({'RapidHeartRate':3 ,'HighHeartRate':2,'NormalHeartRate':1})
    df1["BreathingDifficulty"] = df1["BreathingDifficulty"].map({'HighBreathingDifficulty':3 ,'BreathingDifficulty':2,'NoBreathingDifficulty':1})
    df1["SkinTinge"] = df1["SkinTinge"].map({'Bluish':3 ,'LightBluish':2,'NotBluish':1})
    df1["Responsiveness"] = df1["Responsiveness"].map({'UnResponsive':3 ,'SemiResponsive':2,'Responsive':1})
    df1["Movement"] = df1["Movement"].map({'Diminished':3 ,'Decreased':2,'NormalMovement':1})
    df1["DeliveryType"] = df1["DeliveryType"].map({'C_Section':3 ,'DifficultDelivery':2,'NormalDelivery':1})
    df1["MothersBPHistory"] = df1["MothersBPHistory"].map({'VeryHighBP':3 ,'HighBP':2,'BPInRange':1})
    df1["CardiacArrestChance"] = df1["CardiacArrestChance"].map({'High':2 ,'Medium':1,'Low':0})
    data = df1[["BirthWeight","FamilyHistory","PretermBirth","HeartRate","BreathingDifficulty","SkinTinge","Responsiveness","Movement","DeliveryType","MothersBPHistory","CardiacArrestChance"]].to_numpy()
    inputs = data[:,:-1]
    outputs = data[:, -1]
    training_inputs = inputs[:1000]
    training_outputs = outputs[:1000]
    testing_inputs = inputs[1000:]
    testing_outputs = outputs[1000:]
    classifier = BaggingClassifier()
    imputer = SimpleImputer(strategy='most_frequent')  # You can change the strategy as needed
    training_outputs = imputer.fit_transform(training_outputs.reshape(-1, 1)).ravel()
    classifier.fit(training_inputs, training_outputs)
    predictions = classifier.predict(testing_inputs)
    bc_accuracy = 100.0 * accuracy_score(testing_outputs, predictions)
    return classifier , bc_accuracy

nn , nn_accuracy = train_neural_network()
bc ,bc_accuracy = train_bagging_classifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.to_dict()
    user_input['BirthWeight'] = {'WeightTooLow': 3, 'LowWeight': 2, 'NormalWeight': 1}[user_input['BirthWeight']]
    user_input['FamilyHistory'] = {'AboveTwoCases': 3, 'ZeroToTwoCases': 2, 'NoCases': 1}[user_input['FamilyHistory']]
    user_input['PretermBirth'] = {'4orMoreWeeksEarlier': 3, '2To4weeksEarlier': 2, 'NotaPreTerm': 1}[user_input['PretermBirth']]
    user_input['HeartRate'] = {'RapidHeartRate': 3, 'HighHeartRate': 2, 'NormalHeartRate': 1}[user_input['HeartRate']]
    user_input['BreathingDifficulty'] = {'HighBreathingDifficulty': 3, 'BreathingDifficulty': 2, 'NoBreathingDifficulty': 1}[user_input['BreathingDifficulty']]
    user_input['SkinTinge'] = {'Bluish': 3, 'LightBluish': 2, 'NotBluish': 1}[user_input['SkinTinge']]
    user_input['Responsiveness'] = {'UnResponsive': 3, 'SemiResponsive': 2, 'Responsive': 1}[user_input['Responsiveness']]
    user_input['Movement'] = {'Diminished': 3, 'Decreased': 2, 'NormalMovement': 1}[user_input['Movement']]
    user_input['DeliveryType'] = {'C_Section': 3, 'DifficultDelivery': 2, 'NormalDelivery': 1}[user_input['DeliveryType']]
    user_input['MothersBPHistory'] = {'VeryHighBP': 3, 'HighBP': 2, 'BPInRange': 1}[user_input['MothersBPHistory']]
        # Predict using Bagging Classifier
        # bagging_prediction = bagging_model.predict([[BirthWeight, FamilyHistory, PretermBirth, HeartRate, BreathingDifficulty,
        #                                              SkinTinge, Responsiveness, Movement, DeliveryType, MothersBPHistory]])

        # Predict using Neural Network Cl1assifier
    nn_prediction = nn.predict(np.array([[user_input['BirthWeight'], user_input['FamilyHistory'], user_input['PretermBirth'], 
                                               user_input['HeartRate'], user_input['BreathingDifficulty'], user_input['SkinTinge'], 
                                               user_input['Responsiveness'], user_input['Movement'], user_input['DeliveryType'], 
                                               user_input['MothersBPHistory']]]))
    predicted_class_index = np.argmax(nn_prediction)
    class_labels = ['Low', 'Medium', 'High']
    predicted_class_label = class_labels[predicted_class_index]

    input_array = np.array([[user_input[attr] for attr in df.columns[:-1]]])
    class_label_mapping = {
       0: 'Low',
       1: 'Medium',
       2: 'High'
    }
    prediction_num = bc.predict(input_array)
    prediction_word = class_label_mapping[prediction_num[0]]
        # Calculate accuracy
        # bagging_accuracy = accuracy_score(y_test, bagging_model.predict(X_test))
        # nn_accuracy = accuracy_score(y_test, nn_model.predict(X_test))

        # return render_template('result.html', bagging_prediction=bagging_prediction[0], nn_prediction=nn_prediction[0],
        #                        bagging_accuracy=bagging_accuracy, nn_accuracy=nn_accuracy)
    return render_template('result.html', nn_prediction=predicted_class_label , bagging_prediction=prediction_word , bagging_accuracy=bc_accuracy, nn_accuracy=nn_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
