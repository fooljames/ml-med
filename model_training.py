from myconfig import *
from utils import *

import pandas as pd
from xgboost import XGBClassifier
import pickle

df = pd.read_pickle(output_pkl)

cols = ['Respiratory Rate'
    , 'Mean Airway Pressure'
    , 'Inspired Tidal Volume'
    , 'SpO2'
    , 'Heart Rate'
    , 'Extrinsic PEEP'
    , 'Pulse Rate']

trim_df, trim_features = moving_trim_avg(df, col=cols)
target_df = check_y(trim_df, n_decrease_lower_bound=0.5, delta_change=-2, start=600, end=720)

feature_df, features = create_features(target_df, n_before=3, t_before='300s', cols=cols)
feature_df = feature_df[~feature_df['SpO2'].isnull()]

valid_locations = feature_df.groupby("dataset_location")['y_flag'] \
    .agg({'y_count': np.sum, 'n_row': np.size}).query("y_count > 0 and n_row > 10000").index
feature_df = feature_df[feature_df.dataset_location.isin(valid_locations)]

x_data = feature_df[features].as_matrix()
y_data = feature_df['y_flag'].as_matrix()


def under_sampling(labels, p):
    negative_idx = np.random.choice(np.where(labels == 0)[0], size=int(len(labels) * p), replace=False)
    positive_idx = np.where(labels == 1)[0]
    return np.concatenate([negative_idx, positive_idx])


def get_auc(params):
    auc = []
    best_ntree_limit = []
    clf = XGBClassifier(n_estimators=500)
    clf.set_params(**params)
    for i in range(len(valid_locations)):
        test_loc = [valid_locations[i]]

        x_train = x_data[~feature_df.dataset_location.isin(test_loc)]
        x_test = x_data[feature_df.dataset_location.isin(test_loc)]
        y_train = y_data[~feature_df.dataset_location.isin(test_loc)]
        y_test = y_data[feature_df.dataset_location.isin(test_loc)]

        resampled_idx = under_sampling(y_train, 0.1)
        x_resampled, y_resampled = x_train[resampled_idx], y_train[resampled_idx]

        eval_set = [(x_test, y_test)]
        clf.fit(x_resampled, y_resampled, early_stopping_rounds=20, eval_metric=["auc"], eval_set=eval_set,
                verbose=True)
        auc.append(clf.best_score)
        best_ntree_limit.append(clf.best_ntree_limit)
    return auc, best_ntree_limit


max_iter = 20
res_df = pd.DataFrame()
for n in range(max_iter):
    colsample_bytree = np.random.uniform(0.1, 0.9)
    subsample = np.random.uniform(0.1, 0.9)
    params = dict(colsample_bytree=colsample_bytree, subsample=subsample)
    auc, best_ntree_limit = get_auc(params)
    res_df = res_df.append(pd.DataFrame([[params, np.mean(auc), np.mean(best_ntree_limit)]],
                                        columns=['params', 'auc', 'best_ntree_limit']))

best_df = res_df[res_df.auc == np.max(res_df.auc)]
best_params = best_df.params[0]
best_ntree_limit = best_df.best_ntree_limit[0]

clf = XGBClassifier(n_estimators=best_ntree_limit)
clf.set_params(**best_params)
clf.fit(x_data, y_data, verbose=True)

# save model to file
pickle.dump(clf, open("model.pickle.dat", "wb"))
