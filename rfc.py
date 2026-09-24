import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.model_selection import train_test_split 

df = pd.read_csv("creditcard.csv")

# look at how imbalanced ts dataset is
sizes = df['Class'].value_counts(sort=1)
#print(sizes)

df.drop(['Time'], axis=1, inplace=True)
df.drop(['Amount'], axis=1, inplace=True)
df.drop(df.columns[[25, 24, 4, 22, 23, 21, 5]], axis=1, inplace=True)

# print(df)

# define dependent variable

Y = df['Class'].values

#define independent variables

X = df.drop(['Class'], axis=1)



X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.4, random_state=20)

from sklearn.ensemble import RandomForestClassifier

baseModel = RandomForestClassifier(n_estimators=30, random_state=30)
baseModel.fit(X_train, Y_train)

basePrediction = baseModel.predict_proba(X_test)[:,1]


basePredBinary = (basePrediction >= 0.5).astype(int)
from sklearn import metrics
from sklearn.metrics import confusion_matrix
print ("AUPRC" , metrics.average_precision_score(Y_test, basePredBinary))

#evalurate importance and drop some columns
#importance = pd.Series(baseModel.feature_importances_).sort_values(ascending=False)
# print(importance)


counts_1 = pd.Series(Y).value_counts()
plt.bar(counts_1.index, counts_1.values, color=['skyblue','salmon'])
plt.xticks([0,1], ['Non-Fraud', 'Fraud'])
plt.ylabel("Count")
plt.title("Original Class Distribution")
plt.show()


cm = confusion_matrix(Y_test, basePredBinary)
sns.heatmap(cm, 
       annot=True,
         fmt='g', 
          xticklabels=['Not Fraud','Fraud'],
           yticklabels=['Not Fraud','Fraud'])
plt.ylabel('Actual', fontsize=13)
plt.title('Confusion Matrix before applying smote', fontsize=17, pad=20)
plt.gca().xaxis.set_label_position('top') 
plt.xlabel('Prediction', fontsize=13)
plt.gca().xaxis.tick_top()


plt.show()






