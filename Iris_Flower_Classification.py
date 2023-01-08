import pandas as pd
from sklearn.model_selection import train_test_split   # for splitting the data into traing and testing part
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width','Class_labels']
df= pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\OasisInfoByte\Iris.csv", names = columns)

#separating input and output columns
data= df.values
X= data[: , 0:4]
Y= data[:, 4]

#splitting data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)


#training the model with the training-data
from sklearn.linear_model import LogisticRegression
model_LR= LogisticRegression()
model_LR.fit(X_train, Y_train)

#testing the data with he testing-data
prediction =model_LR.predict(X_test)


#to calculate accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,prediction)*100)

#comparing the predicted and the actual output for the testing data
for i in range(len(prediction)):
    print(Y_test[i], "  ", prediction[i])
