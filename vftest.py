import pandas as pd
import vfdata as vfd
import vfmodel as vfm
import pytest

def test_get_majority_store():
    df = vfd.get_prepared_data()
    assert ((df.loc[(df['prefecture'] == 'Tōkyō-to') &
           (df['city'].isin(['Shibuya-ku', 'Minato-ku'])) &
           (df['city-area'].isin(['Shibuya-ku Shibuya',
                                  'Minato-ku Shibakōen'])) &
           (df['air_genre_name'].isin(['Izakaya', 'Cafe/Sweets'])),
           'air_store_id'].nunique()) ==
           df.loc[vfd.get_majority_store(df), 'air_store_id'].nunique())

def test_is_uniform_expected_less_than_five():
    with pytest.raises(ValueError):
        vfd.is_uniform(pd.Series([4]))

def test_is_uniform_observed_less_than_five():
    with pytest.raises(ValueError):
        vfd.is_uniform(pd.Series([4, 50, 50]))

@pytest.mark.parametrize('test_input,expected', [
    (pd.Series([10, 10]), (True, pd.Series([10, 10]))),
    (pd.Series([10, 11]), (True, pd.Series([10, 11]))),
    (pd.Series([30, 10, 10, 10]), (False, pd.Series([30, 10, 10, 10]))),
    (pd.Series([28, 27, 33, 37]), (True, pd.Series([28, 27, 33, 37]))),
    (pd.Series([30, 10, 10, 5]), (False, pd.Series([30, 10, 10, 5]))),
])
def test_is_uniform(test_input, expected):
    uniform, counts = vfd.is_uniform(test_input)
    assert uniform == expected[0]
    assert (counts.values == expected[1].values).all()

@pytest.mark.parametrize('test_input,expected', [
    (pd.Series([10, 10]), pd.Series([10, 10])),
    (pd.Series([10, 11]), pd.Series([11])),
    (pd.Series([30, 10, 10, 10]), pd.Series([30])),
    (pd.Series([30, 30, 10, 10, 10]), pd.Series([30, 30])),
    (pd.Series([28, 27, 33, 34]), pd.Series([33, 34])),
    (pd.Series([30, 30, 30, 5]), pd.Series([30, 30, 30])),
])
def test_remove_low_counts(test_input, expected):
    test_output = vfd.remove_low_counts(test_input)
    assert len(test_output) == len(expected)
    assert (test_output.values == expected.values).all()

@pytest.mark.parametrize('test_input,expected', [
    (pd.Series([10, 10, 11]), pd.Series([10, 10])),
    (pd.Series([10, 11, 15]), pd.Series([10, 11])),
    (pd.Series([30, 10, 10, 10]), pd.Series([10, 10, 10])),
    (pd.Series([28, 27, 33, 37]), pd.Series([28, 27, 33])),
    (pd.Series([30, 30, 30, 5]), pd.Series([30, 30, 30])),
])
def test_remove_largest_deviation(test_input, expected):
    test_output = vfd.remove_largest_deviation(test_input)
    assert len(test_output) == len(expected)
    assert (test_output.values == expected.values).all()

@pytest.mark.parametrize('test_input,expected', [
    (pd.Series([10]), pd.Series([10])),
    (pd.Series([10, 10]), pd.Series([10, 10])),
    (pd.Series([10, 11]), pd.Series([10, 11])),
    (pd.Series([10, 10, 11]), pd.Series([10, 10, 11])),
    (pd.Series([10, 11, 15]), pd.Series([10, 11, 15])),
    (pd.Series([30, 30, 10, 10, 10]), pd.Series([30, 30])),
    (pd.Series([30, 10, 10, 10]), pd.Series([30])),
    (pd.Series([28, 27, 33, 37]), pd.Series([28, 27, 33, 37])),
    (pd.Series([28, 27, 33, 34]), pd.Series([28, 27, 33, 34])),
    (pd.Series([30, 30, 30, 5]), pd.Series([30, 30, 30])),
    (pd.Series([30, 30, 30, 1]), pd.Series([30, 30, 30])),
])
def test_get_uniform_counts(test_input, expected):
    test_output = vfd.get_uniform_counts(test_input)
    assert len(test_output) == len(expected)
    assert (test_output.values == expected.values).all()

def test_get_uniform_counts_expected_less_than_five():
    with pytest.raises(ValueError):
        vfd.get_uniform_counts(pd.Series([4, 5]))

@pytest.mark.parametrize('test_input,expected', [
    ((pd.Series([1]*20), 5), True),
    ((pd.Series([1]*19), 5), False),
])
def test_is_enough_data_available(test_input, expected):
    df, length = test_input
    assert vfm.is_enough_data_available(df, length) == expected
