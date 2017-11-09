import pandas as pd

location_date = ['dataset_location', 'dataset_datetime']

def mean_by_interval(df, interval):
    """
    mean ignore NULL value
    :param df: input
    :param interval: time interval
    :return: aggregated dataframe
    """

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

    ## drop duplicates
    data = data.drop_duplicates(location_date + list(selected_cols))

    ## get mapping from id to colname
    colnames.index = colnames.id
    name_dict = colnames['name'].to_dict()

    ## change columns into names
    data.columns = location_date + [name_dict[c] for c in selected_cols]

    return data


def separate_type(df):
    # divide data into settings and measurements
    settings = df[location_date + list(df.columns[df.columns.str.contains("Setting")])]
    measurement = df[list(df.columns[~df.columns.str.contains("Setting")])]
    return settings, measurement


def find_consecutive_null(df):
    """
    :param df: df after mean by interval
    :return: Series where index is columns name and value is the corresponding max consecutive null
    """
