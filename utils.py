import pandas as pd
import numpy as np

location_date = ['dataset_location', 'dataset_datetime']


def mean_by_interval(df, interval):
    df = df.copy(deep=True)
    df['datetime_interval'] = pd.to_datetime(df['time_in_sec'] // interval * interval, unit='s')
    df = df.groupby(['datetime_interval', 'dataset_location']).mean().reset_index()
    return df


def load_data(data_path, colname_path):
    """
    change column's name to a be readable one
    measurement -> name_measure
    categorical -> name_cat
    config -> name_config
    :param df: 
    :return: 
    """
    ## load data and colnames files
    data = pd.read_csv(data_path)
    colnames = pd.read_csv(colname_path)

    ## get only relevant columns
    selected_cols = data.columns[data.columns.isin(colnames.id)]
    data = data[location_date + list(selected_cols)]

    ## preprocessing
    data = data.drop_duplicates(location_date + list(selected_cols))
    data = data[np.isfinite(data['dataset_datetime'])]

    ## get mapping from id to colname
    colnames.index = colnames.id
    name_dict = colnames['name'].to_dict()

    ## change columns into names
    data.columns = location_date + [name_dict[c] for c in selected_cols]

    return data


def separate_type(df):
    # divide data into settings and measurements
    settings = df[location_date + list(df.columns[df.columns.str.contains("Setting|Mode")])]
    measurement = df[list(df.columns[~df.columns.str.contains("Setting|Mode")])]
    return settings, measurement


def get_valid_measurement(measurement):
    valid_cols = set(measurement.columns[measurement.nunique() > 100]) - set(location_date)
    return measurement[list(valid_cols)]


def find_consecutive_null(df):
    """
    :param df: df after mean by interval
    :return: Series where index is columns name and value is the corresponding max consecutive null
    """


def addTimeInSec(df):
    df = df.copy(deep=True)
    df['time_in_sec'] = pd.to_datetime(df['dataset_datetime'].astype(int).astype(str), format="%Y%m%d%H%M%S").astype(
        'int64') // 1000000000
    return df


def moving_avg(df):
    df.index = df['dataset_datetime']
    df = df.rename(columns={'dataset_a0188': 'SpO2'})
    f = lambda x: x.rolling('1h').mean()
    df['SpO2_moving_avg'] = df.groupby('dataset_location')['SpO2'].apply(f)
    df['SpO2_percent_change'] = (df['SpO2'] - df['SpO2_moving_avg']) / df['SpO2_moving_avg']
    return df

def check_y(df, t='180s', n_decrease_lower_bound = 6, delta_change = -0.1):
    df.index = df['dataset_datetime']
    df['check'] = np.where(df['SpO2_percent_change'] <= delta_change, 1, 0)
    f = lambda x: x.rolling(t).sum()
    df['y_check_decrease'] = df.groupby('dataset_location')['check'].apply(f)
    setting_cols = [col for col in demographic.columns if 'setting' in col or 'mode' in col]
    df['is_setting_changed'] =df[setting_cols].rolling('1h', center = True).std().max().max() > 0
    df['y_value'] = np.where((df['check'] == 1) & (df['y_check_decrease'] >= n_decrease_lower_bound) & df['is_setting_changed'], 1, 0)
    del df['check']
    df.index = range(len(df))
    return df
