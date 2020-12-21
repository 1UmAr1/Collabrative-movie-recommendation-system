import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, RepeatedKFold
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("Book1.csv", encoding="cp850")
data.dropna(axis=0)
lab = pd.read_csv("labels.csv")
lab.dropna(axis=0)
X_train, X_test, y_train, y_test = train_test_split(data, lab, test_size=0.3)

clf = RandomForestRegressor(n_estimators=100, max_features='auto', oob_score=True)
cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
n_scores = cross_val_score(clf, X_train, y_train, scoring=make_scorer(metrics.mean_squared_error), cv=cv, n_jobs=1, error_score="raise")
print("SCOORE:::: %3f (%.3f)" % (np.mean(n_scores), np.std(n_scores)))

clf.fit(X_train, y_train)
feature = pd.Series(clf.feature_importances_, index=data.columns).sort_values(ascending=False)
print(feature)

sns.barplot(x=feature, y=feature.index)
plt.xlabel("Feature importance Score")
plt.ylabel("Features")
plt.title("Visualizing Importance Feature")
plt.legend()
plt.show()

pre = clf.predict(data)


