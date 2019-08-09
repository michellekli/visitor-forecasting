import numpy as np
import pandas as pd
import pmdarima as pm
import pydlm
import statsmodels.api as sm
import math
import warnings
warnings.filterwarnings('ignore')
import vfdata as vfd
import vfconstants as vfc

def fit_arima(data, **kwargs):
    return pm.auto_arima(data, exogenous=None,
                         start_p=0, d=None, start_q=0, max_p=7, max_d=2, max_q=7,
                         seasonal=False, stationary=False,
                         information_criterion='aicc',
                         alpha=0.05, test='kpss',
                         stepwise=True,
                         error_action='ignore')

def fit_arima_results(data, arima_order=(1,0,0), **kwargs):
    return sm.tsa.ARIMA(data, arima_order).fit()

def fit_predict_arima(data, forecast_horizon, arima_order=(1,0,0), **kwargs):
    results = fit_arima_results(data, arima_order=arima_order)
    predictions = results.forecast(forecast_horizon, alpha=0.05)
    return predictions[0], predictions[2], results.resid

def fit_sarimax(data, sarimax_m=5, **kwargs):
    return pm.auto_arima(data, exogenous=None,
                     start_p=0, d=None, start_q=0, max_p=7, max_d=1, max_q=7,
                     m=sarimax_m, start_P=0, D=None, start_Q=0, max_P=2, max_D=1, max_Q=2,
                     seasonal=True, stationary=False,
                     information_criterion='aicc',
                     alpha=0.05, test='kpss',
                     stepwise=True,
                     error_action='ignore')

def fit_sarimax_results(data, sarimax_order=(1,0,0), sarimax_seasonal_order=(0,0,0), **kwargs):
    results = sm.tsa.statespace.SARIMAX(
        data,
        order=sarimax_order,
        trend='c',
        seasonal_order=sarimax_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False).fit()
    return results

def fit_predict_sarimax(data, forecast_horizon, sarimax_order=(1,0,0), sarimax_seasonal_order=(0,0,0), **kwargs):
    results = fit_sarimax_results(
        data,
        sarimax_order=sarimax_order,
        sarimax_seasonal_order=sarimax_seasonal_order)
    forecast = results.get_forecast(steps=forecast_horizon)
    return forecast.predicted_mean, forecast.conf_int(alpha=0.05), results.resid

def fit_bsts(data, bsts_order=(1,0,0), bsts_seasonal_order=(0,0,0), **kwargs):
    ar, _, _ = bsts_order
    _, _, _, period = bsts_seasonal_order

    model = pydlm.dlm(data) + pydlm.trend(degree=0, discount=0.95, name='trend_const', w=10)
    if ar > 0:
        model = model + pydlm.autoReg(degree=ar, discount=0.95, name='ar', w=10)
    if period > 0:
        model = model + pydlm.seasonality(period=period, discount=0.99, name='week', w=1e7)
    model.fit()

    return model

def fit_bsts_results(data, bsts_order=(1,0,0), bsts_seasonal_order=(0,0,0), **kwargs):
    class Results:
        # class to mimic the Results object returned by statsmodels
        def __init__(self, data, bsts_order, bsts_seasonal_order):
            self.data = data
            self.bsts_order = bsts_order
            self.bsts_seasonal_order = bsts_seasonal_order
            # todo: how to get bsts predictions on training data
            self.model = fit_bsts(data, bsts_order=bsts_order, bsts_seasonal_order=bsts_seasonal_order)
            self.fittedvalues = np.array(self.model.getAll().predictedObs).flatten()

        def forecast(self, forecast_horizon):
            forecast, _, _ = fit_predict_bsts(data,
                forecast_horizon,
                bsts_order=bsts_order,
                bsts_seasonal_order=bsts_seasonal_order)
            return forecast

    results = Results(data, bsts_order, bsts_seasonal_order)

    return results

def fit_predict_bsts(data, forecast_horizon, bsts_order=(1,0,0), bsts_seasonal_order=(0,0,0), **kwargs):
    model = fit_bsts(data, bsts_order=bsts_order, bsts_seasonal_order=bsts_seasonal_order)
    forecast, variance = model.predictN(date=model.n-1, N=forecast_horizon)
    std_dev = pd.Series(variance).apply(math.sqrt)
    lower = (forecast - 2*std_dev)
    upper = (forecast + 2*std_dev)
    return np.array(forecast), np.array([lower, upper]).transpose(), np.array(model.getResidual())

def is_enough_data_available(df, season_length):
    return df.shape[0] >= (4 * season_length)

def train_validation_test_split(df, forecast_horizon):
    train = df[:-(2*forecast_horizon)]
    test = df[df.shape[0]-forecast_horizon:]
    validation = df[train.shape[0]:train.shape[0]+forecast_horizon]

    assert forecast_horizon == test.shape[0]
    assert forecast_horizon == validation.shape[0]
    assert train.shape[0] + test.shape[0] + validation.shape[0] == df.shape[0]
    assert (train.append(validation.append(test)).index == df.index).all()

    return train, validation, test

def get_forecast_horizon(window_end_date, forecast_horizon, day_names):
    forecast_data = pd.DataFrame(pd.date_range(start=(pd.to_datetime(window_end_date) +
                                                     pd.DateOffset(days=1)),
                                               end=(pd.to_datetime(window_end_date) +
                                                     pd.DateOffset(days=forecast_horizon))),
                                 columns=['visit_date'])
    forecast_data['visit_day_name'] = forecast_data['visit_date'].apply(lambda x: x.day_name())
    forecast_data = forecast_data.loc[forecast_data['visit_day_name'].isin(day_names)]

    return forecast_data.shape[0]

def get_sle(forecast, actual, missing=None):
    # squared log error for entire forecast horizon
    sle = ((np.log(np.array(forecast) + 1) - np.log(np.array(actual) + 1)) ** 2)
    if missing is not None:
        # missing used as a mask here, but 1 indicates the value is missing
        # so need to flip values from 0 to 1 and 1 to 0
        mask = missing ^ np.ones(len(missing), dtype=int)
        return np.sum(sle * mask)
    return np.sum(sle)

def get_rmsle(results):
    return results.groupby(by='model')['forecast_error'].apply(lambda x: np.sqrt(np.mean(x)))

def get_arima_window_size(train, validation, missing, order, season_length):
    best_window_size = len(train)
    best_sle = np.inf
    errors = []
    for i in range(0, len(train) - (4*season_length)):
        results = fit_arima_results(train[i:], order)
        sle = get_sle(results.forecast(len(validation))[0], validation, missing)
        errors.append(sle)
        if sle <= best_sle:
            best_sle = sle
            best_window_size = len(train) - i
    return best_window_size, pd.Series(errors[::-1], index=range(4*season_length, len(train)))

def get_sarimax_window_size(train, validation, missing, order, seasonal_order):
    warnings.filterwarnings('ignore')

    best_window_size = len(train)
    best_sle = np.inf
    errors = []
    for i in range(0, len(train) - (4*seasonal_order[3])):
        results = fit_sarimax_results(train[i:], order, seasonal_order)
        sle = get_sle(results.forecast(len(validation)), validation, missing)
        errors.append(sle)
        if sle < best_sle:
            best_sle = sle
            best_window_size = len(train) - i

    return best_window_size, pd.Series(errors[::-1], index=range(4*seasonal_order[3], len(train)))

def get_window_size(train, validation, missing, model, season_length, **kwargs):
    warnings.filterwarnings('ignore')

    best_window_size = len(train)
    best_sle = np.inf
    errors = []
    for i in range(0, len(train) - (4*season_length)):
        predictions, _, _ = eval('fit_predict_{}(train[i:], len(validation), **kwargs)'.format(model))
        sle = get_sle(predictions, validation, missing)
        errors.append(sle)
        if sle < best_sle:
            best_sle = sle
            best_window_size = len(train) - i

    return best_window_size, pd.Series(errors[::-1], index=range(4*season_length, len(train)))

def rolling_window_analysis(index, train, missing, model, window_size=28, forecast_horizon=1, **kwargs):
    if len(index) != len(train):
        raise ValueError('`index` and `train` must be the same length')

    warnings.filterwarnings('ignore')

    visit_dates, outcomes, residuals, errors, forecasts, previews, models = ([] for i in range(7))
    for visit_date, i in zip(index[window_size:], range(0, len(index) - window_size - forecast_horizon + 1)):
        window = train[i:i+window_size]
        actual = train[i+window_size:i+window_size+forecast_horizon]
        missing_mask = missing[i+window_size:i+window_size+forecast_horizon]
        if model in vfc.SUPPORTED_MODELS:
            predictions, _, resids = eval('fit_predict_{}(window, forecast_horizon, **kwargs)'.format(model))
            visit_dates.append(visit_date)
            outcomes.append(actual)
            previews.append(predictions[0])
            forecasts.append(predictions)
            errors.append(get_sle(predictions, actual, missing_mask))
            residuals.append(resids)
            models.append('{}_{}_{}'.format(model, window_size, forecast_horizon))
        else:
            raise ValueError('model should be one of: all | {}'.format(' | '.join(vfc.SUPPORTED_MODELS)))

    results = pd.DataFrame(index=visit_dates)
    results['observed'] = outcomes
    results['forecast_error'] = errors
    results['fitted_residual'] = residuals
    results['forecast'] = forecasts
    results['preview'] = previews
    results['model'] = models

    return results

def all_model_rolling_window_analysis(index, train, missing, params, **kwargs):
    results = []
    lengths = []
    for model in params:
        results.append(rolling_window_analysis(index, train, missing, model, **params[model], **kwargs))
        lengths.append(results[-1].shape[0])

    shortest = pd.Series(lengths).idxmin()
    for i in range(len(lengths)):
        if i != shortest:
            results[shortest] = results[shortest].append(results[i][-lengths[shortest]:])
    return results[shortest]

def evaluate_model(train, test, missing, model, **kwargs):
    forecast, _, _ = eval('fit_predict_{}(train, test.shape[0], **kwargs)'.format(model))
    return get_sle(forecast, test, missing)

def get_best_model(analysis, train, test, missing, params):
    test_sle = {}
    lowest_sle = np.inf
    best_model = None
    best_window_size = None
    for mwf in analysis['model'].unique():
        model, window_size, _ = mwf.split('_')
        test_sle[model] = evaluate_model(
            train[-params[model]['window_size']:],
            test,
            missing,
            model,
            **params[model])
        total_sle = analysis.loc[analysis['model']==mwf, 'forecast_error'].sum() + test_sle[model]
        if total_sle < lowest_sle:
            lowest_sle = total_sle
            best_model = model
            best_window_size = window_size

    return best_model, best_window_size, test_sle

def make_forecast(full_df, train, model, forecast_horizon, day_names, **kwargs):
    warnings.filterwarnings('ignore')

    window_size = kwargs['window_size']
    window = train[-window_size:]
    if model in vfc.SUPPORTED_MODELS:
        forecast, conf_int, _ = eval('fit_predict_{}(window, forecast_horizon, **kwargs)'.format(model))
        results = eval('fit_{}_results(window, **kwargs)'.format(model))
    else:
        raise ValueError('model should be one of: all | {}'.format(' | '.join(vfc.SUPPORTED_MODELS)))

    full_df['fitted'] = (pd.Series([np.NaN] * (full_df.shape[0] - window_size))
                               .append(pd.Series(results.fittedvalues)).values)
    assert (full_df['fitted'][-1:] == results.fittedvalues[-1]).all()

    # create dataframe with only the forecast and confidence intervals
    forecast_data = pd.DataFrame(pd.date_range(start=(pd.to_datetime(full_df['visit_date'].values[-1]) +
                                                     pd.DateOffset(days=1)),
                                               end=(pd.to_datetime(full_df['visit_date'].values[-1]) +
                                                     pd.DateOffset(days=forecast_horizon))),
                                 columns=['visit_date'])
    forecast_data['visit_day_name'] = forecast_data['visit_date'].apply(lambda x: x.day_name())
    forecast_data['forecast'] = forecast
    assert (forecast_data['forecast'] == forecast).all()

    forecast_data = forecast_data.join(
        pd.DataFrame(conf_int,
                     index=forecast_data.index,
                     columns=['95_lower', '95_upper']))
    assert (forecast_data['95_lower'].values == conf_int[:,0]).all()
    assert (forecast_data['95_upper'].values == conf_int[:,1]).all()

    forecast_data = forecast_data.loc[forecast_data['visit_day_name'].isin(day_names)]
    assert forecast_data.shape[0] == get_forecast_horizon(full_df['visit_date'].values[-1], forecast_horizon, day_names)

    forecast_data['fitted'] = np.NaN
    full_df = full_df.append(forecast_data, sort=False).reset_index(drop=True)

    return full_df

def forecast_for_store(store, forecast_horizon):
    audit = {}
    audit['store_id'] = store
    audit['forecast_horizon'] = forecast_horizon

    df = vfd.get_prepared_data()
    store_df, day_names = vfd.get_prepared_stable_store_data(df, store)
    audit['day_names'] = day_names
    audit['nrows'] = store_df.shape[0]

    if ((store_df.shape[0] - forecast_horizon*2) <
        get_forecast_horizon(
            store_df.loc[store_df['visit_day_name'].isin(day_names)].index.values[-1],
            forecast_horizon,
            day_names)):
        raise ValueError('{} rows for training. Not enough data available. '
                         'Please try a shorter forecast horizon. '
                         .format(store_df.shape[0] - forecast_horizon*2))

    train, validation, test = train_validation_test_split(store_df, forecast_horizon)
    audit['train'] = train
    audit['validation'] = validation
    audit['test'] = test

    # if train.shape[0] + validation.shape[0] < 60:
    #     raise ValueError('Not enough data available. '
    #                      'Please try a shorter forecast horizon.')
    # assert is_enough_data_available(train, len(day_names))

    sarimax_model = fit_sarimax(
        train.loc[train['visit_day_name'].isin(day_names), 'visitors'],
        sarimax_m=len(day_names))

    sarimax_window_size, sarimax_window_errors = get_window_size(
        train.loc[train['visit_day_name'].isin(day_names), 'visitors'].values,
        validation.loc[validation['visit_day_name'].isin(day_names), 'visitors'].values,
        validation.loc[validation['visit_day_name'].isin(day_names), 'missing'].values,
        'sarimax',
        len(day_names),
        sarimax_order=sarimax_model.order,
        sarimax_seasonal_order=sarimax_model.seasonal_order)

    bsts_window_size, bsts_window_errors = get_window_size(
        train.loc[train['visit_day_name'].isin(day_names), 'visitors'].values,
        validation.loc[validation['visit_day_name'].isin(day_names), 'visitors'].values,
        validation.loc[validation['visit_day_name'].isin(day_names), 'missing'].values,
        'bsts',
        len(day_names),
        bsts_order=sarimax_model.order,
        bsts_seasonal_order=sarimax_model.seasonal_order)

    params = {
        'sarimax': {
            'window_size': sarimax_window_size,
            'sarimax_order': sarimax_model.order,
            'sarimax_seasonal_order': sarimax_model.seasonal_order,
        },
        'bsts': {
            'window_size': bsts_window_size,
            'bsts_order': sarimax_model.order,
            'bsts_seasonal_order': sarimax_model.seasonal_order,
        },
    }
    audit['params'] = params

    train_full = train.append(validation)

    assert train_full.shape[0] == train.shape[0] + validation.shape[0]
    assert train.index[0] == train_full.index[0]
    assert validation.index[-1] == train_full.index[-1]

    analysis = all_model_rolling_window_analysis(
        train_full.loc[train_full['visit_day_name'].isin(day_names)].index,
        train_full.loc[train_full['visit_day_name'].isin(day_names), 'visitors'].values,
        train_full.loc[train_full['visit_day_name'].isin(day_names), 'missing'].values,
        params,
        forecast_horizon=get_forecast_horizon(
            train_full.loc[train_full['visit_day_name'].isin(day_names)].index.values[-1],
            forecast_horizon,
            day_names))
    audit['rolling_window_analysis'] = analysis

    best_model, best_window_size, best_test_sle = get_best_model(
        analysis,
        train_full.loc[train_full['visit_day_name'].isin(day_names), 'visitors'].values,
        test.loc[test['visit_day_name'].isin(day_names), 'visitors'].values,
        test.loc[test['visit_day_name'].isin(day_names), 'missing'],
        params)
    audit['best_model'] = best_model
    audit['best_window_size'] = best_window_size
    audit['best_test_sle'] = best_test_sle

    forecast = make_forecast(
        store_df.loc[store_df['visit_day_name'].isin(day_names)].reset_index(),
        store_df.loc[store_df['visit_day_name'].isin(day_names), 'visitors'].values,
        best_model,
        forecast_horizon,
        day_names,
        **params[best_model])
    return forecast, audit
