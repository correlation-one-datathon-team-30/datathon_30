# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
# Load pandas
import pandas as pd

# Load numpy
import numpy as np

data = pd.read_csv('u_14_g_14_15_s_0.01.csv')
#print(data.iloc[0])
data = data._get_numeric_data()
Xp = data.drop(['isuber'],axis=1)
X = Xp.as_matrix().astype(np.float64)
y = data['isuber'].as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X.shape[1]), Xp.columns[indices])
plt.xlim([-1, X.shape[1]])
plt.show()