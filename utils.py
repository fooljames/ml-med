import pandas as pd
import numpy as np

location_date = ['dataset_location', 'dataset_datetime']


def mean_by_interval(df, interval):
    df = df.copy(deep=True)
    df = df.groupby([df['time_in_sec'] // interval, df['dataset_location']]).mean()
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
    valid_cols = set(measurement.columns[measurement.nunique() > 1]) - set(location_date)
    return measurement[valid_cols]


def find_consecutive_null(df):
    """
    :param df: df after mean by interval
    :return: Series where index is columns name and value is the corresponding max consecutive null
    """
    
def addTimeInSec (df):
    df = df.copy(deep=True)
    hhmmss = df['dataset_datetime'].astype(str)
    df['datetime'] = pd.to_datetime(df['dataset_datetime'].astype(int).astype(str), format="%Y%m%d%H%M%S")
    df['time_in_sec'] = df['datetime'].astype('int64')//1000000000
    return df
