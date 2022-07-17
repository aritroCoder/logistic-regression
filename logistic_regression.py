#train a logistic regression classifier to predict whether a flower is iris virginica or not

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

#basic notes on numpy slicing
# Slicing in python means taking elements from one given index to another given index.

# We pass slice instead of index like this: [start:end].

# We can also define the step, like this: [start:end:step].

# If we don't pass start its considered 0

# If we don't pass end its considered length of array in that dimension

# If we don't pass step its considered 1
# Note: The result includes the start index, but excludes the end index.

X = iris["data"][:, 3:] #take all the rows and only fourth coloumn
Y = (iris["target"]==2).astype(np.int) #this will be a boolean array that contain only true/false values and the classifier will be a binary classifier. Noter that target == 2 stands for iris virginica and astype function converts the boolean array to integer array(0 is false and 1 is true).

#train a logistic regression classifier.
clf = LogisticRegression()
clf.fit(X, Y)
example = clf.predict([[2.6]])
print(example)

#using matplotlib to plot visulatization
#In NumPy, -1 in reshape(-1) refers to an unknown dimension that the reshape() function calculates for you.
# It is like saying: “I will leave this dimension for the reshape() function to determine”. Syntax of reshape is reshape(row, column)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1) #get 1000 pts b/w 0 and 3 by using linspace and reshape it to a column vector. Reshape changes it to one coloumn and any number of rows
Y_prob = clf.predict_proba(X_new) #predict the probability of the class for use in logistic regression.
plt.plot(X_new, Y_prob[:, 1], "g-", label="virginica")
plt.show()