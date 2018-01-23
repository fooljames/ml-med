import pandas as pd
import numpy as np
import glob

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
    df['dataset_datetime'] = pd.to_datetime(df['time_in_sec'] // interval * interval, unit='s')
    df = df.groupby(['dataset_datetime', 'dataset_location']).mean().reset_index()
    return df


def moving_avg(df, t = '600s'):
    df.index = df['dataset_datetime']
    rolling_mean = df.groupby('dataset_location')['SpO2'].rolling(t).mean().reset_index()
    rolling_mean.rename(columns={'SpO2': 'SpO2_moving_avg'}, inplace=True)
    df = pd.merge(df, rolling_mean, on=['dataset_location', 'dataset_datetime'])
    # df['SpO2_moving_avg'] = df.groupby('dataset_location')['SpO2'].rolling('600s').mean().reset_index(0,drop=True)
    df['SpO2_percent_change'] = (df['SpO2'] - df['SpO2_moving_avg']) / df['SpO2_moving_avg']
    df.index = range(len(df))
    return df


# merge capsule and phlilp file ato be output file
def merge_file(phlilps_path, capsule_path, output_path):
    phlilps_list = glob.glob(phlilps_path + "/phlilps*.csv")
    capsule_list = glob.glob(capsule_path + "/capsule*.csv")
    output_list = glob.glob(output_path + "/output*.csv")
    output_file = [x[46:54] for x in output_list]

    for i, p in enumerate(phlilps_list):
        date_file = p.split('-')[1][:8]
        if date_file in output_file:
            print('output_%s already existed' % date_file)

        else:
            for j, c in enumerate(capsule_list):
                if date_file == c.split('-')[1][:8]:
                    phlilps = load_data(p, phlilps_path + '/column_id_name_p.csv')
                    phlilps = addTimeInSec(phlilps)
                    phlilps = mean_by_interval(phlilps, 10)
                    phlilps = moving_avg(phlilps)

                    if int(date_file) < 20171124:
                        phlilps['ward'] = np.where(phlilps['dataset_location'].str[3:4] == '1', '7', '6')
                        phlilps['dataset_location1'] = phlilps['dataset_location']
                        phlilps['dataset_location'] = 'M' + phlilps['dataset_location1'].str[:4] + '-' + phlilps[
                            'ward'] + 'FL-B' + phlilps['dataset_location1'].str[5:6]
                        del phlilps['ward']
                        del phlilps['dataset_location1']

                    capsules = load_data(c, capsule_path + '/column_id_name_c.csv')
                    capsules = addTimeInSec(capsules)
                    capsules = mean_by_interval(capsules, 10)
                    del capsules['time_in_sec']

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


# check y
def check_y(df, t='180s', n_decrease_lower_bound=0.8, delta_change=-3.0):
    df['SpO2_change'] = df['SpO2'] - df['SpO2_moving_avg']
    df.sort_values(by=['dataset_location', 'dataset_datetime'], inplace=True)
    df.index = df['dataset_datetime']
    df['check'] = np.where(df['SpO2_change'] <= delta_change, 1, 0)
    summ = lambda x: x.rolling(t).sum()
    count = lambda x: x.rolling(t).count()
    df['data_count'] = df.groupby('dataset_location')['check'].apply(summ)
    df['y_check_decrease'] = df.groupby('dataset_location')['check'].apply(count) / df['data_count']
    df['y_value'] = np.where(df['SpO2'].isnull() == True, 'NA', 0)
    df = df[df['y_value'] != 'NA']
    df['y_value'] = np.where(
        (df['check'] == 1) & (df['y_check_decrease'] >= n_decrease_lower_bound) & (df['data_count'] > int(t[:3]) / 20),
        1, df['y_value'])  # & (df['is_setting_changed'])
    df['sum_y_value'] = df.groupby('dataset_location')['y_value'].apply(summ)
    df['y_cut_flag'] = np.where((df['y_value'] == 1) & (df['sum_y_value'] > 1), 'cut', df['y_value'])
    df['y_value'] = np.where((df['y_value'] == 1) & (df['sum_y_value'] == 1), 1, 0)


    # if within 5-10 mins. ahead has 1, then 1
    sum5 = lambda x: x.rolling('300s').sum()
    sum10 = lambda x: x.rolling('600s').sum()
    df['data_y_5m'] = df.groupby('dataset_location')['y_value'].apply(sum5)
    df['data_y_10m'] = df.groupby('dataset_location')['y_value'].apply(sum10)
    df['y_diff'] = df['data_y_10m'] - df['data_y_5m']
    df['y_flag'] = np.where(df['y_diff'] > 0, 1, 0)
    del df['data_y_5m']
    del df['data_y_10m']
    del df['y_diff']
    df.index = range(len(df))
    return df


# create features
def create_features(df, t_moving='180s', t_before='600s', n_before=6, cols = ['Respiratory Rate', 'Mean Airway Pressure', 'Inspired Tidal Volume', 'SpO2']):
    data = df.copy()

    # Moving Average ############################################################################################
    data.sort_values(by=['dataset_location', 'dataset_datetime'], inplace=True)
    data.index = data['dataset_datetime']
    mean = data.groupby('dataset_location')[cols].rolling(t_moving).mean().reset_index()
    mean.rename(columns = {col: '{}_{}'.format(col, 'moving_mean_avg') for col in (cols)}, inplace = True)
    sd = data.groupby('dataset_location')[cols].rolling(t_moving).std().reset_index()
    sd.rename(columns = {col: '{}_{}'.format(col, 'moving_sd_avg') for col in (cols)}, inplace = True)

    data = pd.merge(data, mean, how = 'left', on = ['dataset_location', 'dataset_datetime'])
    data = pd.merge(data, sd, how='left', on = ['dataset_location', 'dataset_datetime'])

    # Average Before ###########################################################################################
    from datetime import timedelta
    data.sort_values(by=['dataset_location', 'dataset_datetime'], inplace=True)
    data.index = data['dataset_datetime']
    # mean and sd 10 mins before
    mean_bf = data.groupby(['dataset_location'])[cols].rolling(t_before).mean().reset_index()
    sd_bf = data.groupby(['dataset_location'])[cols].rolling(t_before).std().reset_index()

    # create list of time ex t_before = 600s >> time_delta = [600, 1200, 1800, 2400, ..., 600*(n_before+1)]
    time_delta = []
    for i in range(0, n_before):
        time_delta.append(int(t_before[0:3]) * i)

    for i, s in enumerate(time_delta):
        # create new column: datetime-time_delta
        col_name = 'datetime_' + str(s) + "s_bf"
        data[col_name] = data['dataset_datetime'] - timedelta(seconds=s)

        # mean
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
        del data[col_name]
    data.index = range(len(data))
    return data


def remove_non_ascii(s):
    return "".join(i for i in s if (ord(i) < 128) & (ord(i) > 47))


# load first admit and follow up
def first_follow(path_first, path_follow):
    first_admit = pd.read_csv(path_first, low_memory=False, index_col=None, encoding='iso-8859-1')
    from datetime import timedelta
    sevenhour = timedelta(seconds=25200)
    first_admit['ICU_ADMIT_DATE_1'] = pd.to_datetime(first_admit['ICU_ADMIT_DATE'].astype(str), format='%Y-%m-%d',
                                                     errors='coerce')
    first_admit['HOSPITAL_ADMIT_DATE_1'] = pd.to_datetime(first_admit['HOSPITAL_ADMIT_DATE'].astype(str),
                                                          format='%Y-%m-%d', errors='coerce')
    first_admit['TIMEATICU_1'] = pd.to_datetime(first_admit['TIMEATICU'].astype(str).str[0:8], format='%H:%M:%S',
                                                errors='coerce').dt.time
    first_admit['ICU_ADMIT_DATETIME'] = pd.to_datetime(
        first_admit['ICU_ADMIT_DATE_1'].dt.strftime('%Y-%m-%d') + ' ' + first_admit['TIMEATICU_1'].astype(str),
        format='%Y-%m-%d %H:%M:%S', errors='coerce') + sevenhour
    first_admit['HOSPITAL_ADMIT_DATE_DATETIME'] = pd.to_datetime(
        first_admit['HOSPITAL_ADMIT_DATE_1'].dt.strftime('%Y-%m-%d') + ' ' + first_admit['TIMEATICU_1'].astype(str),
        format='%Y-%m-%d %H:%M:%S', errors='coerce') + sevenhour
    first_admit['datetime'] = np.where(first_admit['ADMITFROM'] == 1, first_admit['ICU_ADMIT_DATETIME'],
                                       first_admit['HOSPITAL_ADMIT_DATE_DATETIME'])

    follow_up = pd.read_csv(path_follow, low_memory=False, index_col=None, encoding='iso-8859-1')
    follow_up['_submission_time'] = pd.to_datetime(follow_up['_submission_time'], format='%Y-%m-%dT%H:%M:%S')

    col1 = ['HN2', 'BED2', 'WARD2', '_submission_time']
    follow_up_s = pd.DataFrame(follow_up, columns=col1)
    follow_up_s = follow_up_s.rename(
        columns={'HN2': 'HN', 'BED2': 'Bed', 'WARD2': 'Ward', 'STATUS': 'Status', '_submission_time': 'datetime'})
    follow_up_s['datasource'] = 'follow_up'

    col2 = ['HN1', 'BED1', 'WARD1', 'datetime']
    first_admit_s = pd.DataFrame(first_admit, columns=col2)
    first_admit_s = first_admit_s.rename(
        columns={'HN1': 'HN', 'BED1': 'Bed', 'WARD1': 'Ward', 'GENDER': 'Gender', 'ICU_ADMIT_DATETIME': 'datetime'})
    first_admit_s['datasource'] = 'first_admit'

    form = pd.concat([follow_up_s, first_admit_s], ignore_index=True)
    form['datetime'] = pd.to_datetime(form['datetime'].dt.strftime('%Y-%m-%d %H:%M'), format='%Y-%m-%d %H:%M')
    form['dataset_location1'] = 'MICU1-' + form['Ward'].astype(str) + 'FL-B' + form['Bed'].astype(str)
    form['dataset_location2'] = 'MICU2-' + form['Ward'].astype(str) + 'FL-B' + form['Bed'].astype(str)
    form['dataset_location'] = np.where((form['Ward'].astype(str) == '7'), form['dataset_location1'].str.strip(),
                                        form['dataset_location2'].str.strip())
    form['dataset_location'] = np.where((form['Ward'] == 'n/a') | (form['Bed'] == 'n/a'), 'n/a',
                                        form['dataset_location'].str.strip())
    del form['dataset_location1']
    del form['dataset_location2']

    ### Clean HN Data
    form['HN'] = form.HN.str.lower()
    form['HN'] = form.HN.apply(remove_non_ascii)
    form['Bed'] = form.Bed.astype(str).str.replace("'", '')
    form['Ward'] = form.Ward.astype(str).str.replace("'", '')

    print('already concated first admit and follow up')
    return form


# define patient
def patient(df, first_admit_path, follow_up_path):
    file = first_follow(first_admit_path, follow_up_path)
    df.sort_values(by=['dataset_location', 'dataset_datetime'], inplace=True)
    df_test = df.copy()

    df_test['HN'] = float('nan')
    for i, x in enumerate(file.HN.unique()):
        location = file[(file.HN == x) & (file.dataset_location != 'n/a')]['dataset_location'].unique()

        print('patient id: %s' % (x))

        for j, y in enumerate(location):
            mindate = file[(file.HN == x) & (file.dataset_location == y)]['datetime'].min()
            maxdate = file[(file.HN == x) & (file.dataset_location == y)]['datetime'].max()

            df_test['HN'] = np.where(((df_test['dataset_datetime'] >= mindate) &
                                      (df_test['dataset_datetime'] <= maxdate) &
                                      (df_test['dataset_location'] == y))
                                     , x, df_test['HN'])

            print('location: %s since %s to %s' % (y, mindate, maxdate))
        print('-------------------------------------------------------------------------------------------------')
    return df_test

