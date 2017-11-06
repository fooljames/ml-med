def mean_by_interval(df, interval):
    """
    mean ignore NULL value
    :param df: input
    :param interval: time interval
    :return: aggregated dataframe
    """


def filter_duplicate(df):
    """
    filter out duplicated rows
    :param df: 
    :return: 
    """


def change_col_names(df):
    """
    change column's name to a be readable one
    measurement -> name_measure
    categorical -> name_cat
    config -> name_config
    :param df: 
    :return: 
    """


def find_consecutive_null(df):
    """
    :param df: df after mean by interval
    :return: Series where index is columns name and value is the corresponding max consecutive null
    """
