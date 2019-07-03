import pandas as pd
import numpy as np


def create_index(data, start=0):

    """
    Creates a sequential mapping for a list of unique IDs.

    ...

    Parameters
    ----------
    data : array-like
        list or array of unique IDs to be encoded sequentially

    Returns
    -------
    encoder : dictionary
        mapping from original data to index

    decoder : dictionary
        mapping from index to original data
    """

    encoder = {}
    decoder = {}

    idx = start
    for item in data:

        if item not in encoder.keys():
            encoder[item] = idx
            decoder[idx] = item
            idx += 1

    return encoder, decoder


def reset_ids(df, column, start=0):

    """
    Returns dataframe with column IDs reset sequentially between 0-n
    """

    ids = np.sort(df[column].unique())
    encoder, _ = create_index(ids)

    df[column] = df[column].apply(lambda x: encoder[x])

    return df


def load_wide(path, user_thresh=4, item_thresh=4, reindex=True):

    """
    Load interactions data in rectangular ratings matrix format.

    A ratings matrix is (usually) mostly sparse, with one row for every user
    and one columns for every item/product. The values in the cells represent
    the ratings for each user-item interactions.

    ...

    Parameters
    ----------
    path : string
        relative path to csv file

    user_thresh : int
        users with fewer interactions than this number will be ommitted

    item_thresh : int
        items with fewer interactions than this number will be omitted

    reindex : bool
        if True, rename columns as a sequential list, starting at 1

    Returns
    -------
    wide_df : Pandas DataFrame
        DataFrame with first column for user_id, other columns refer to item_id.
        Values are ratings for each user-item interaction.
    """

    # read data from file
    wide_df = pd.read_csv(path, index_col=False)
    # assign index name and shift all user IDs by 1
    wide_df.index.name = 'user_id'
    wide_df.reset_index(inplace=True)
    # wide_df.user_id += 1

    # drop columns with no ratings
    wide_df.dropna(axis=1, thresh=item_thresh, inplace=True)
    # drop rows with no ratings
    wide_df.dropna(axis=0, thresh=(user_thresh+1), inplace=True)

    # re-index column names, starting from 1
    if reindex:
        encoder, _ = create_index(wide_df.columns[1:])
        wide_df.columns = ['user_id'] + list(map(str, encoder.values()))

    return wide_df


def make_long(wide_df, users, items, ratings, explicit=True):

    """
    Convert interactions data from wide to long format.

    long_df will contain three columns for users, items and ratings for explicit
    data and two columns for users and items for implicit data.


    Args:
    -----
    csv_file (string):
        Path to the csv file with user-item interactions

    users (string):
        column name from wide_df for user IDs

    items (string):
        name to be assigned to the items column of df

    ratings (string):
        name to be assigned to the ratings column of df

    Returns
    -------
    long_df (Pandas DataFrame):
        DataFrame with columns 'user_id', 'item_id' and 'rating'
    """

    # melt table from wide to long
    long_df = wide_df.melt(id_vars=users, var_name=items, value_name=ratings)

    # drop rows with NaN ratings
    long_df.dropna(inplace=True)

    # for implicit data
    if not explicit:
        long_df.drop(ratings, axis=1, inplace=True)

    return long_df


def make_wide(long_df, vals, idx, cols):

    """
    Convert interactions data from long to wide format.

    See `load_wide()` for more information on this format.

    ...

    Parameters
    ----------
    long_df : Pandas DataFrame
        interaction data in long format

    vals : string
        column name from long_df for ratings

    idx : string
        column name from long_df for user IDs

    cols : string
        column name from long_df for item IDs

    Returns
    -------
    wide_df : Pandas DataFrame
        DataFrame with first column for user_id, other columns refer to item_id.
        Values are ratings for each user-item interaction.
    """

    # pivot table from long to wide
    wide_df = pd.pivot_table(long_df, values=vals, index=idx, columns=cols)

    # reorder columns
    col_values = wide_df.columns.values.astype('int64')
    order = np.sort(col_values).astype('str')
    wide_df.columns = wide_df.columns.astype('str')
    wide_df = wide_df[order]

    # reset index
    wide_df.reset_index(inplace=True)
    del wide_df.columns.name

    return wide_df
