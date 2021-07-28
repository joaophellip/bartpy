from bartpy.sklearnmodel import SklearnModel
from sklearn.model_selection import GridSearchCV
import pandas as pd

data = pd.read_csv('creditcard.csv')

model = SklearnModel(n_burn=500, n_chains=1, n_jobs=1, n_samples=500, n_trees=10)
X = data.drop('Class', axis=1).iloc[:1001, ]
y = data['Class'].iloc[:1001, ]
parameters = {'n_trees': [10]}
model.fit(X, y)

#grid_search = GridSearchCV(model, parameters)
# print(grid_search.best_params_)
