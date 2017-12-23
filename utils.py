import pandas as pd
import numpy as np

location_date = ['dataset_location', 'dataset_datetime']


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
    colnames = pd.read_csv(colname_path, low_memory=False)

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


def addTimeInSec(df):
    df = df.copy(deep=True)
    df['time_in_sec'] = pd.to_datetime(df['dataset_datetime'].astype(str), format="%Y%m%d%H%M%S").astype(
        'int64') // 1000000000
    return df


def mean_by_interval(df, interval = 10):
    df = df.copy(deep=True)
    df['datetime_interval'] = pd.to_datetime(df['time_in_sec'] // interval * interval, unit='s')
    df = df.groupby(['datetime_interval', 'dataset_location']).mean().reset_index()
    return df


def moving_avg(df, t = '600s'):
    df.index = df['dataset_datetime']
    rolling_mean = df.groupby('dataset_location')['SpO2'].rolling(t).mean().reset_index()
    rolling_mean.rename(columns={'SpO2': 'SpO2_moving_avg'}, inplace=True)
    df = pd.merge(df, rolling_mean, on=['dataset_location', 'dataset_datetime'])
    # df['SpO2_moving_avg'] = df.groupby('dataset_location')['SpO2'].rolling('600s').mean().reset_index(0,drop=True)
    df['SpO2_percent_change'] = (df['SpO2'] - df['SpO2_moving_avg']) / df['SpO2_moving_avg']
    df.index = range(len(df))


def merge_file(phlilps_path, capsule_path, output_path):
    phlilps_list = glob.glob(phlilps_path + "/phlilps*.csv")
    capsule_list = glob.glob(capsule_path + "/capsule*.csv")
    output_list = glob.glob(output_path + "/output*.csv")
    output_file = [x[46:54] for x in output_list]

    for i, p in enumerate(phlilps_list):
        date_file = p[47:55]
        if date_file in output_file:
            print('output_%s already existed' % date_file)

        else:
            for j, c in enumerate(capsule_list):
                if date_file == c[48:56]:
                    phlilps = load_data(p, phlilps_path + '/column_id_name_p.csv')
                    # phlilps = mean_interval(phlilps)
                    phlilps = addTimeInSec(phlilps)
                    phlilps = mean_by_interval(phlilps, 10)
                    ##phlilps = y_moving_avg(phlilps)
                    del phlilps['diff_time']

                    capsules = load_data(c, capsule_path + '/column_id_name_c.csv')
                    capsules = addTimeInSec(capsules)
                    capsules = mean_by_interval(capsules, 10)

                    df = pd.merge(left=capsules, right=phlilps, how='inner',
                                  on=['dataset_datetime', 'dataset_location'])
                    # df.to_hdf(output_path + '/output_' + date_file + '.h5', index = False, key = 'output')
                    df.to_csv(output_path + '/output_' + date_file + '.csv', index=False)
                    print("already saved file %s" % date_file)


def load_data_output(datapath):
    # file_list = glob.glob(datapath + "/output*.h5")
    file_list = glob.glob(datapath + "/output*.csv")
    data = pd.DataFrame()
    list_ = []
    for file_ in file_list:
        if file_ == file_list[0]:
            df = pd.read_csv(file_, low_memory=False, index_col=None)
        else:
            df = pd.read_csv(file_, low_memory=False, index_col=None, header=0)
        data = data.append(df)
        print("Loaded file %s with rows: %d " % (file_, len(df)))

    data.index = range(len(data))
    data = data.drop_duplicates()
    data['dataset_datetime'] = pd.to_datetime(data['dataset_datetime'], format='%Y-%m-%d %H:%M:%S', )
    return data

def check_y(df, t='180s', n_decrease_lower_bound=6, delta_change=-0.1):
    df.sort_values(by=['dataset_location', 'dataset_datetime'], inplace=True)
    df.index = df['dataset_datetime']
    df['check'] = np.where(df['SpO2_percent_change'] <= delta_change, 1, 0)
    f = lambda x: x.rolling(t).sum()
    df['y_check_decrease'] = df.groupby('dataset_location')['check'].apply(f)

    # df.columns.str.lower()
    # setting_cols = [col for col in df.columns if 'setting' in col or 'mode' in col]
    # df['is_setting_changed'] = df[setting_cols].rolling('3600s', center = True).std().max().max() > 0
    df['y_value'] = np.where(df['SpO2'].isnull() == True, 'NA', 0)
    df['y_value'] = np.where((df['check'] == 1) & (df['y_check_decrease'] >= n_decrease_lower_bound), 1,
                             df['y_value'])  # & (df['is_setting_changed'])
    del df['check']
    df.index = range(len(df))
    return df


def create_features(df, t_moving='180s', t_before='600s', n_before=6):
    data = df.copy()
    cols = ['Respiratory Rate', 'Mean Airway Pressure', 'Inspired Tidal Volume', 'SpO2']

    # Moving Average ############################################################################################
    data.sort_values(by=['dataset_location', 'dataset_datetime'], inplace=True)
    data.index = data['dataset_datetime']
    mean = data.groupby('dataset_location')[cols].rolling(t_moving).mean().reset_index()
    # mean.rename(columns = {col: '{}_{}'.format(col, 'moving_mean_avg') for col in (cols)}, inplace = True)
    sd = data.groupby('dataset_location')[cols].rolling(t_moving).std().reset_index()
    # sd.rename(columns = {col: '{}_{}'.format(col, 'moving_sd_avg') for col in (cols)}, inplace = True)

    for i, col in enumerate(cols):
        colname = col + "_moving_mean_avg"
        data[colname] = mean[col]
        colname = col + "_moving_sd_avg"
        data[colname] = sd[col]

    # Average Before ###########################################################################################
    from datetime import timedelta
    mean_bf = data.groupby(['dataset_location'])[cols].rolling(t_before).mean().reset_index()
    sd_bf = data.groupby(['dataset_location'])[cols].rolling(t_before).std().reset_index()

    time_delta = []
    for i in range(1, n_before + 1):
        time_delta.append(int(t_before[0:3]) * i)

    for i, s in enumerate(time_delta):
        col_name = 'datetime_' + str(s) + "s_bf"
        data[col_name] = data['dataset_datetime'] - timedelta(seconds=s)

        mean_df = mean_bf.copy()
        mean_df = mean_df.rename(columns={'dataset_datetime': col_name})
        mean_df.rename(columns={col: '{}_{}'.format(col, 'mean' + str(s) + 's') for col in (cols)}, inplace=True)
        data = pd.merge(left=data, right=mean_df, how='left', left_on=[col_name, 'dataset_location'],
                        right_on=[col_name, 'dataset_location'])

        std_df = sd_bf.copy()
        std_df = std_df.rename(columns={'dataset_datetime': col_name})
        std_df.rename(columns={col: '{}_{}'.format(col, 'std' + str(s) + 's') for col in (cols)}, inplace=True)
        data = pd.merge(left=data, right=std_df, how='left', left_on=[col_name, 'dataset_location'],
                        right_on=[col_name, 'dataset_location'])

    data.index = range(len(data))
    return data


def patient(df):
    df.sort_values(by = ['dataset_location', 'dataset_datetime'], inplace = True)
    df_test = df.copy()
    df_test['HN'] = float('nan')
    for i, x in enumerate(file.HN.unique()):
        location = file[(file.HN == x) & (file.dataset_location != 'n/a')]['dataset_location'].unique()
        print( 'patient id: %s' %(x))

        for j, y in enumerate(location):
            mindate = file[(file.HN == x) & (file.dataset_location == y) ]['datetime'].min()
            maxdate = file[(file.HN == x) & (file.dataset_location == y)]['datetime'].max()

            df_test['HN'] = np.where(((df_test['dataset_datetime'] >= mindate) &
                                     (df_test['dataset_datetime'] <= maxdate) &
                                     (df_test['dataset_location'] == y))
                                     , x, df_test['HN'])

            print('location: %s since %s to %s' %(y, mindate, maxdate))
        print('-------------------------------------------------------------------------------------------------')
    return df_test

