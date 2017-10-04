from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# Modeling


### Only CV
gkf = GroupKFold(n_splits=3).split(df_train[features], df_train[label], df_train['OF_Bobine'])
model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=14, min_samples_split=10)
scores = cross_val_score(model, df_train[features], df_train[label], cv=gkf)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

### CV and grid-search
gkf = GroupKFold(n_splits=3).split(df_train[features], df_train[label], df_train['OF_Bobine'])
parameters = {'max_depth': (2, 3, 4, 5, 6)}
dtc = DecisionTreeClassifier(min_samples_leaf=14, min_samples_split=10)
model = GridSearchCV(dtc, parameters, cv=gkf, scoring='f1_weighted')
model.fit(df_train[features], df_train[label])
print("::: Best params found: {}".format(model.best_params_))
for s in model.grid_scores_:
    print(s)



