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


def merge_all(capsule, phlilp, first_admit, follow_up):
    df_capsule = capsule
    df_phlilp = phlilp
    first_admit = first_admit
    follow_up = follow_up

    """
    merge capsule and phlilp
    """
    data = pd.merge(left=df_phlilp, right=df_capsule, how='right', on=['dataset_datetime', 'dataset_location'])
    data['Ward'] = pd.to_numeric(data['dataset_location'].str[6:7])
    data['Bed'] = pd.to_numeric(data['dataset_location'].str[11:12])

    """
    merge first_admit and follow_up
    """
    first_admit['ICU_ADMIT_DATE_1'] = pd.to_datetime(first_admit['ICU_ADMIT_DATE'].astype(str), format='%m/%d/%Y',
                                                     errors='coerce')
    first_admit['TIMEATICU_1'] = pd.to_datetime(first_admit['TIMEATICU'].astype(str).str[0:8], format='%H:%M:%S',
                                                errors='coerce').dt.time
    first_admit['ICU_ADMIT_DATETIME'] = pd.to_datetime(
        first_admit['ICU_ADMIT_DATE_1'].dt.strftime('%Y-%m-%d') + ' ' + first_admit['TIMEATICU_1'].astype(str),
        format='%Y-%m-%d %H:%M:%S', errors='coerce') + sevenhour
    del first_admit['ICU_ADMIT_DATE_1'], first_admit['TIMEATICU_1']

    follow_up['_submission_time'] = pd.to_datetime(follow_up['_submission_time'], format='%Y-%m-%dT%H:%M:%S')

    # concat first_admit andfollow_up
    col1 = ['HN2', 'BED2', 'WARD2', 'STATUS', '_submission_time']
    follow_up_s = pd.DataFrame(follow_up, columns=col1)
    follow_up_s = follow_up_s.rename(
        columns={'HN2': 'HN', 'BED2': 'Bed', 'WARD2': 'Ward', 'STATUS': 'Status', '_submission_time': 'datetime'})
    follow_up_s['datasource'] = 'follow_up'

    col2 = ['HN1', 'BED1', 'WARD1', 'GENDER', 'ICU_ADMIT_DATETIME']
    first_admit_s = pd.DataFrame(first_admit, columns=col2)
    first_admit_s = first_admit_s.rename(
        columns={'HN1': 'HN', 'BED1': 'Bed', 'WARD1': 'Ward', 'GENDER': 'Gender', 'ICU_ADMIT_DATETIME': 'datetime'})
    first_admit_s['datasource'] = 'first_admit'

    form = pd.concat([follow_up_s, first_admit_s], ignore_index=True)

    """
    merge all
    """
    data['datetime'] = pd.to_datetime(data_s['dataset_datetime'].dt.strftime('%Y-%m-%d %H:%M'), format='%Y-%m-%d %H:%M')
    form['datetime'] = pd.to_datetime(form['datetime'].dt.strftime('%Y-%m-%d %H:%M'), format='%Y-%m-%d %H:%M')

    # join on Ward, Bed and datetime (miniute level)
    df = pd.merge(left=data, right=form, how='left', left_on=['Ward', 'Bed', 'datetime'],
                  right_on=['Ward', 'Bed', 'datetime'])
    df = df.sort_values(by=['Bed', 'Ward', 'datetime'])

    # for loop with each HN and get mindate, maxdate, bed and ward
    # and then assign HN to other rows with condition: same bed, same ward, datetime in (mindate,maxdate)
    for i, p in enumerate(df['HN'].unique()):
        mindate = df[df['HN'] == p]['dataset_datetime'].min()
        maxdate = df[df['HN'] == p]['dataset_datetime'].max()
        Bed = df[df['HN'] == p]['Bed'].min()
        Ward = df[df['HN'] == p]['Ward'].min()
        Gender = df[df['HN'] == p]['Gender'].min()

        df['HN'] = np.where(((df['dataset_datetime'] >= mindate) & (df['dataset_datetime'] <= maxdate) & (
        df['Bed'] == Bed) & (df['Ward'] == Ward)), p, df['HN'])
        df['Gender'] = np.where(((df['dataset_datetime'] >= mindate) & (df['dataset_datetime'] <= maxdate) & (
        df['Bed'] == Bed) & (df['Ward'] == Ward)), Gender, df['Gender'])

    del df['datetime']
    return df
