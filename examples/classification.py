from bartpy.classifiersklearnmodel import ClassifierSklearnModel
import pandas as pd
from joblib import dump, load

try:
    restored_model = load("first_model.joblib")
    print(restored_model)
except:
    pass

data_df = pd.read_csv('creditcard.csv')
indexes_df = pd.read_csv('indexes.csv')

idx = indexes_df[indexes_df['ds'] == 'training']['index'].values - 1    # indexes generated in R - starting in 1

model = ClassifierSklearnModel(n_burn=1000, n_chains=1, n_jobs=1, n_samples=2000, n_trees=10, alpha=0.95, beta=2.0)
X = data_df.drop('Class', axis=1).iloc[idx, ]
y = data_df['Class'].iloc[idx, ]

model.fit(X, y)

dump(model, 'first_model.joblib')

val_idx = indexes_df[indexes_df['ds'] == 'validation']['index'].values - 1
val_x = data_df.drop('Class', axis=1).iloc[val_idx, ]
val_y = data_df['Class'].iloc[val_idx, ]

predictions = model.predict(val_x)

print(f"top-1 accuracy: {(predictions == val_y).sum() / val_y.shape[0]}")
