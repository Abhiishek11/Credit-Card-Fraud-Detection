# Credit Card Fraud Detection

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing and visualizing the dataset
dataset = pd.read_csv('creditcard.csv')
print(dataset.columns) #Shows us the colums of the dataset
print(dataset.shape) #Gives us the total Entries in the dataset and the total number of columns
print(dataset.describe()) #Gives us the mean sd etc of each column
dataset.hist(figsize= (20,20))
plt.show()#Plots the histogram of the given dataset

Fraudcases = dataset[dataset['Class']==1] #Number of fraud cases in the dataset
Validcases = dataset[dataset['Class']==0] #Number of Valid Cases in the dataset

Total_fraction = len(Fraudcases)/float(len(Validcases)) #Calculates the fraction of committing a fraud

print(Total_fraction)
print('Total Fraud Cases:{}'.format(len(Fraudcases)))
print('Total Valid Cases:{}'.format(len(Validcases)))

#Building a correlation TO remove extra things from our dataset
matrix = dataset.corr()
figure = plt.figure(figsize = (12,12))
sns.heatmap(matrix, vmax=0.7, square = True)
plt.show()

#Filtering the Dataset and removing the data which we dont need
columns = dataset.columns.tolist()
columns = [c for c in columns if c not in['Class']]

target = "Class"

X = dataset[columns]
y = dataset[target]

print(X.shape)
print(y.shape)


# Importing the libraries for algorithm
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score



# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


#Predicting probability
probs = classifier.predict_proba(X_test)
print(probs)
#Keeping outcomes of only positive probabiity
probs = probs[:, 1]

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)

#Calculating the Total Score/Accuracy of the model
score = classifier.score(X_test,y_test)
print("Model Evaluated")
print(score)



#Plotting the Precesion Recall Curves

# Calculate the Precesion-Recall Curve 
precision, recall, thresholds = precision_recall_curve(y_test, probs)

# Calculate the F1 Score
f1 = f1_score(y_test, y_pred)

# Calculate Precesion-Recall AUC
auc = auc(recall, precision)

# Calculate Average Precesion Score
ap = average_precision_score(y_test, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))


#  Plot the Precesion Recall Curve for the model
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
plt.plot(recall, precision, marker='.')
plt.show()