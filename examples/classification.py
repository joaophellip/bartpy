from numpy.random.mtrand import f
from bartpy.classifiersklearnmodel import ClassifierSklearnModel
import pandas as pd
from joblib import dump, load

SKIP_TRAINING = False
SKIP_VALIDATION = False
EXPORT_MODEL = False

try:
    restored_model = load("debug_model.joblib")
    print(restored_model)
    CACHED_MODEL = True
except:
    CACHED_MODEL = False
    pass

data_df = pd.read_csv('creditcard.csv')
indexes_df = pd.read_csv('indexes.csv')

if not SKIP_TRAINING or not CACHED_MODEL:

    idx = indexes_df[indexes_df['ds'] == 'training']['index'].values - 1  # indexes generated in R - starting in 1
    model = ClassifierSklearnModel(n_burn=1000, n_chains=1, n_jobs=1, n_samples=2000, n_trees=10, alpha=0.95, beta=2.0)
    x = data_df.drop(['Class', 'Time'], axis=1).iloc[idx[4900:5010], ]
    y = data_df['Class'].iloc[idx[4900:5010], ]

    model.fit(x, y)

    if EXPORT_MODEL:
        dump(model, 'debug_model.joblib')

else:
    model = restored_model

if not SKIP_VALIDATION:

    val_idx = indexes_df[indexes_df['ds'] == 'validation']['index'].values - 1
    val_x = data_df.drop(['Class', 'Time'], axis=1).iloc[[5], ]
    val_y = data_df['Class'].iloc[[5], ]

    print(f"class label is {val_y.values}")

    prob_pred = model.predict(val_x)

    print(f"prob X is a fraud: {prob_pred}")

    #binary_pred = [1 if x > 0.5 else 0 for x in prob_pred]
    #print(f"top-1 accuracy: {(binary_pred == val_y).sum() / val_y.shape[0]}")

else:
    print("validation skipped.")
