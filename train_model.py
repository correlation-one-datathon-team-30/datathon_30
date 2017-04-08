# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
# Load pandas
import pandas as pd

# Load numpy
import numpy as np

data = pd.read_csv('full_0.01.csv')
#print(data.iloc[0])
data = data._get_numeric_data()
data.drop(['latitude','longitude'],axis=1,inplace=True)
Xp = data.drop(['isuber'],axis=1)
X = np.array(Xp)[:,1:].astype(np.float64)
y = np.array(data['isuber'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier(n_estimators=30, max_depth=None,min_samples_split=2, random_state=0, n_jobs=4)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices],
#        color="r", align="center")
# plt.xticks(range(X.shape[1]), np.array(list(Xp.columns)[1:])[indices])
# plt.xlim([-1, X.shape[1]])
#plt.show()

predictions = list(clf.predict_proba(X_test))
# indices = [i for i,x in enumerate(predictions) if 0.45<x[0]<0.55]
# print(indices)
# df = pd.DataFrame(X_test[indices],columns=list(Xp.columns)[1:])
# df['isuber'] = y_test[indices]
# df.to_csv('boundary_cases.csv')
import random
#predictions = [p for p in predictions if random.random()<0.05]
print(len(predictions))
hist, bins = np.histogram(predictions, bins=20)
plt.xlabel('Prediction')
plt.title('Histogram of Predictions:')
plt.grid(True)
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()