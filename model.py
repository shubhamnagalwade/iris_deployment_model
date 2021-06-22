import pandas as pd

df = pd.read_csv('iris.csv')
#print(df.head())

#splitting out data into features and target 

X = df.loc[:, df.columns != 'Species']
#print(X.head())
y = df['Species']
#print(y.head())

#split data into train and test 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)

#print(X_train.shape)
#print(X_train.head())
#print(X_test.shape)
#print(X_test.head())
#print(y_train.shape)
#print(y_train.head())
#print(y_test.shape)
#print(y_test.head())

#model creation

from sklearn.ensemble import RandomForestClassifier
#create object of RandomForestClassifier 
model = RandomForestClassifier()

#training 

#train model
model.fit(X_train, y_train)

#print score
#print(model.score(X_train,y_train))

#prediction

#predict X_test data
predictions = model.predict(X_test)
#print(predictions[:10])


#scoring

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#print(accuracy_score(y_test, predictions))
#print(confusion_matrix(y_test, predictions))
#print(classification_report(y_test, predictions))

#save model in output directory
import joblib

joblib.dump(model,'randomforest_model.pkl')

#predict with new data
import numpy as np
test_data = [5.1, 3.2, 1.5, 0.4, 5.2]
#convert test_data into numpy array
test_data = np.array(test_data)
#reshape
test_data = test_data.reshape(1,-1)
#print(test_data)

#load trained model

#declare path where you saved your model

filePath = 'randomforest_model.pkl'
#open file
file = open(filePath, "rb")
#load the trained model
trained_model = joblib.load(file)


#predict with trained data

prediction = trained_model.predict(test_data)
print(prediction)