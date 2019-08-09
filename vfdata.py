import numpy as np
import pandas as pd
from scipy.stats import chisquare
import ruptures as rpt

def get_stores_before(df, date='2016-07-01'):
    return df.loc[df['visit_date'] < date, 'air_store_id'].unique()

def get_stores_starting(df, date='2016-07-01'):
    return set(df['air_store_id'].unique()).difference(set(get_stores_before(df, '2016-07-01')))

def get_stores_before_df(df, date='2016-07-01'):
    return df.loc[df['air_store_id'].isin(get_stores_before(df)), :]

def get_stores_starting_df(df, date='2016-07-01'):
    return df.loc[df['air_store_id'].isin(get_stores_starting(df)), :]

def get_majority_store(df):
    return ((df['prefecture'] == 'Tōkyō-to') &
            (df['city'].isin(['Shibuya-ku', 'Minato-ku'])) &
            (df['city-area'].isin(['Shibuya-ku Shibuya', 'Minato-ku Shibakōen'])) &
            (df['air_genre_name'].isin(['Izakaya', 'Cafe/Sweets'])))

def get_least_data_store_id():
    return 'air_c66dbd2c37832d00'

def get_most_data_store_id():
    return 'air_5c817ef28f236bdf'

def is_uniform(counts):
    # observed and expected frequencies should be at least 5
    if counts.mean() < 5:
        raise ValueError('Expected frequency should be at least 5. The expected frequency is: {}'.format(counts.mean()))
    if len(counts[counts < 5].index) > 0:
        raise ValueError('Observed frequency should be at least 5. The observed frequencies are: {}'.format(counts))
    if len(counts) == 1:
        return True, counts

    return chisquare(counts[counts >= 5])[1] >= 0.05, counts[counts >= 5]

def remove_largest_deviation(counts):
    distance = [np.abs(count - counts.mean()) for count in counts]
    return counts.drop(counts.index[distance.index(max(distance))])

def remove_low_counts(counts):
    if counts.std() == 0:
        # no deviaton means they are all the same value
        return counts

    return counts[counts > counts.mean()]

def get_uniform_counts(day_counts):
    # check for uniformity with all counts
    try:
        uniform, counts = is_uniform(day_counts)
        if uniform:
            return counts
    except ValueError as err:
        if 'Observed frequency should be at least 5' in str(err):
            # this exception will be handled by remove_low_counts
            pass
        else:
            raise err

    # check for uniformity without lowest counts
    uniform, counts = is_uniform(remove_low_counts(day_counts))
    if uniform:
        return counts
    # check for uniformity while removing largest deviation
    while len(counts) > 0:
        counts = remove_largest_deviation(counts)
        uniform, counts = is_uniform(counts)
        if uniform:
            return counts

def get_prepared_stable_store_data(df, store):
    store_df, day_names = get_prepared_store_data(df, store)

    visitors = store_df.loc[store_df['visit_day_name'].isin(day_names)]['visitors']
    model = 'l2'
    algo = rpt.BottomUp(model=model).fit(visitors.values)
    # breakpoints are in sorted order
    my_bkps = algo.predict(n_bkps=2)
    # get end of rupture
    rupture_end = visitors.index[my_bkps[1]]
    # most recent data up to rupture
    stable_df = store_df.loc[store_df.index > rupture_end]

    return stable_df, day_names

def get_prepared_store_data(df, store):
    store_df = df.loc[df['air_store_id']==store, ['visit_date', 'visitors']]
    gapless_df = (pd.DataFrame(index=pd.date_range(start=store_df['visit_date'].min(), end=store_df['visit_date'].max()))
                  .merge(store_df, how='left', left_index=True, right_on='visit_date'))

    assert (gapless_df.loc[gapless_df['visit_date'].isin(store_df['visit_date']), 'visitors'].values ==
            store_df['visitors'].values).all()
    if store_df.shape[0] < gapless_df.shape[0]:
        assert gapless_df.loc[~gapless_df['visit_date'].isin(store_df['visit_date']), :].shape[0] > 0
        assert (np.isnan(gapless_df.loc[~gapless_df['visit_date'].isin(store_df['visit_date']), 'visitors'].values)).all()

    gapless_df.loc[:, 'missing'] = 0
    gapless_df.loc[np.isnan(gapless_df['visitors']).values, 'missing'] = 1

    gapless_df['visit_day_name'] = gapless_df['visit_date'].apply(lambda x: x.day_name())
    day_names = list(get_uniform_counts(gapless_df[gapless_df['missing']==0].groupby(by='visit_day_name').size()).index)
    if len(day_names) == 0:
        # no days with enough counts for modeling
        return None
    gapless_df = gapless_df.loc[gapless_df['visit_day_name'].isin(day_names), :]
    gapless_df = gapless_df.set_index('visit_date')

    # todo: find best way to interpolate time series data
    return gapless_df.interpolate(method='time'), day_names

def get_prepared_data():
    # loading data from csv files
    air_visit_data_df = pd.read_csv('data/visitor-forecasting/air_visit_data.csv')
    air_store_info_df = pd.read_csv('data/visitor-forecasting/air_store_info.csv')
    df = air_visit_data_df.merge(air_store_info_df, on='air_store_id', how='inner')
    assert df.shape[0] == air_visit_data_df.shape[0]
    assert df.shape[1] == (air_visit_data_df.shape[1] + air_store_info_df.shape[1] - 1)

    # casting to timestamp
    df['visit_date'] = pd.to_datetime(df['visit_date'], format='%Y-%m-%d')
    assert type(df.iloc[0]['visit_date']) == pd.Timestamp
    assert df.iloc[0]['visit_date'].year == 2016
    assert df.iloc[0]['visit_date'].month == 1
    assert df.iloc[0]['visit_date'].day == 13

    # sort the data frame by air_store_id and visit_date
    df = df.sort_values(by=['air_store_id', 'visit_date']).reset_index(drop=True)

    # fix areas with more than one pair of coordinates
    unique_area_names = df.groupby(by=['air_area_name']).nunique()
    multiple_coords = unique_area_names.loc[(unique_area_names['latitude'] > 1) | (unique_area_names['longitude'] > 1), :]
    duplicate_areas = df.loc[df['air_area_name'].isin(multiple_coords.index),
           ['air_area_name', 'latitude', 'longitude']].groupby(by=['air_area_name', 'latitude', 'longitude']).size().reset_index()
    for area in duplicate_areas['air_area_name'].unique():
        # get first coordinate pair for that area
        first_result = duplicate_areas.loc[duplicate_areas['air_area_name'] == area].iloc[0]
        # replace all coordinates for that area with first coordinate pair
        df.loc[df['air_area_name'] == area, 'latitude'] = first_result['latitude']
        df.loc[df['air_area_name'] == area, 'longitude'] = first_result['longitude']
    for area in duplicate_areas['air_area_name'].unique():
        assert ((df.loc[df['air_area_name'] == area,
                        ['latitude', 'longitude']].groupby(by=['latitude', 'longitude']).size().shape[0]) == 1)

    # separate prefecture, city, and area from air_area_name
    regions = df['air_area_name'].str.split()
    df['prefecture'] = regions.apply(lambda x: x[0])
    df['city'] = regions.apply(lambda x: x[1])
    df['area'] = regions.apply(lambda x: ' '.join(x[2:]))
    assert (df['air_area_name'] == df['prefecture'] + ' ' + df['city'] + ' ' + df['area']).all()

    # use city-area combination because there are area names repeated across cities
    df['city-area'] = df['city'] + ' ' + df['area']
    df = df.drop('area', axis=1)

    return df
