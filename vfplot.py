import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
import calendar
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import vfdata as vfd

def plot_config(title=''):
    plt.xticks(rotation=45, horizontalalignment='right')
    fig = plt.gcf()
    fig.set_size_inches(16, 10)
    ax = plt.gca()
    ax.set_title(title)

def plot_open_stores_daily(df, stores_subset=None, title=''):
    if stores_subset is None:
        subset_df = df
    else:
        subset_df = df.loc[df['air_store_id'].isin(stores_subset), :]

    subset_counts = subset_df['visit_date'].value_counts().reset_index()
    subset_counts['visit_dow'] = subset_counts['index'].apply(lambda x: x.weekday())
    subset_counts['visit_day_name'] = subset_counts['index'].apply(lambda x: x.day_name())
    subset_counts['color'] = subset_counts['visit_dow'].apply(lambda x: sns.color_palette()[x])

    plt.scatter(subset_counts['index'], subset_counts['visit_date'], c=subset_counts['color'])
    plt.legend(handles=[mpatches.Patch(color=sns.color_palette()[i], label=calendar.day_name[i]) for i in range(7)])
    plot_config(title)
    ax = plt.gca()
    ax.set_ylabel('Number of Restaurants Open')
    plt.show()

    return subset_counts

def plot_median_visitors_daily(df, stores_subset=None, title=''):
    if stores_subset is None:
        subset_df = df
    else:
        subset_df = df.loc[df['air_store_id'].isin(stores_subset), :]

    subset_counts = subset_df.groupby(by='visit_date')['visitors'].median().reset_index()
    subset_counts['visit_dow'] = subset_counts['visit_date'].apply(lambda x: x.weekday())
    subset_counts['visit_day_name'] = subset_counts['visit_date'].apply(lambda x: x.day_name())
    subset_counts['color'] = subset_counts['visit_dow'].apply(lambda x: sns.color_palette()[x])

    sns.lineplot(x='visit_date', y='visitors', hue='visit_day_name', data=subset_counts)
    plot_config(title)
    plt.show()

    return subset_counts

def plot_store_counts_by(df, by, stores_subset=None, title=''):
    if stores_subset is None:
        subset_df = df
    else:
        subset_df = df.loc[df['air_store_id'].isin(stores_subset), :]

    subset_counts = subset_df.groupby(by=by)['air_store_id'].nunique().sort_values(ascending=False)

    sns.barplot(x=subset_counts.index, y=subset_counts, color=sns.color_palette()[0])
    plot_config(title)
    plt.show()
    return subset_counts

def plot_median_visitors_by(df, by, stores_subset=None, title=''):
    if stores_subset is None:
        subset_df = df
    else:
        subset_df = df.loc[df['air_store_id'].isin(stores_subset), :]

    subset_counts = subset_df.groupby(by=by)['visitors'].median().sort_values(ascending=False)

    sns.barplot(x=subset_counts.index, y=subset_counts, color=sns.color_palette()[0])
    plot_config(title)
    plt.show()
    return subset_counts

def plot_store_counts_comparison(df, by, title=''):
    nrows = 2
    ncols = 2

    ax = plt.subplot2grid((nrows,ncols), (1,0), colspan=ncols)
    subset_counts = df.groupby(by=by)['air_store_id'].nunique().sort_values(ascending=False)
    overall_index = list(subset_counts.index)
    sns.barplot(x=subset_counts.index, y=subset_counts, ax=ax)
    plot_config()
    plt.ylabel('Number of Stores')
    plt.title('All Stores')

    ax = plt.subplot2grid((nrows,ncols), (0,0))
    subset_df = df.loc[df['air_store_id'].isin(vfd.get_stores_before(df)), :]
    subset_counts = subset_df.groupby(by=by)['air_store_id'].nunique().sort_values(ascending=False).reindex(index=overall_index)
    sns.barplot(x=subset_counts.index, y=subset_counts, ax=ax)
    plot_config()
    plt.xticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Stores Before July 2016')

    ax = plt.subplot2grid((nrows,ncols), (0,1))
    subset_df = df.loc[df['air_store_id'].isin(vfd.get_stores_starting(df)), :]
    subset_counts = subset_df.groupby(by=by)['air_store_id'].nunique().sort_values(ascending=False).reindex(index=overall_index)
    sns.barplot(x=subset_counts.index, y=subset_counts, ax=ax)
    plot_config()
    plt.xticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Stores Starting July 2016')

    fig = plt.gcf()
    fig.suptitle(title)
    plt.show()

def plot_median_visitors_comparison(df, by, title=''):
    nrows = 1
    ncols = 1

    ax = plt.subplot2grid((nrows,ncols), (0,0))
    subset_counts = df.groupby(by=by)['visitors'].median().sort_values(ascending=False)
    overall_index = list(subset_counts.index)
    sns.scatterplot(x=np.arange(len(overall_index)), y=subset_counts, ax=ax, label='All Stores')
    plot_config()
    plt.xticks(ticks=np.arange(len(overall_index)), labels=overall_index)

    offset = max(1, len(overall_index)//10) / -10
    subset_df = df.loc[df['air_store_id'].isin(vfd.get_stores_before(df)), :]
    subset_counts = subset_df.groupby(by=by)['visitors'].median().sort_values(ascending=False).reindex(index=overall_index)
    sns.scatterplot(x=np.arange(len(overall_index)) + offset, y=subset_counts, ax=ax, label='Before July 2016')
    plot_config()

    offset = max(1, len(overall_index)//10) / 10
    subset_df = df.loc[df['air_store_id'].isin(vfd.get_stores_starting(df)), :]
    subset_counts = subset_df.groupby(by=by)['visitors'].median().sort_values(ascending=False).reindex(index=overall_index)
    sns.scatterplot(x=np.arange(len(overall_index)) + offset, y=subset_counts, ax=ax, label='Starting July 2016')
    plot_config(title)

    plt.show()

def plot_acf_pacf(df=None, index=None, values=None, lags=25, title='Original values'):
    if (index is None or values is None) and df is None:
        raise ValueError("Either both or neither of `index` and `values` must "
                        "be specified.")

    if df is not None:
        index = df.index
        values = df['visitors']

    plt.plot(index, values)
    plot_config(title=title)
    fig = plt.gcf()
    fig.set_size_inches(16, 3)
    ax = plt.gca()
    ax.axhline(y=0, color='gray')
    plt.show()

    plot_acf(values, lags=lags)
    fig = plt.gcf()
    fig.set_size_inches(16, 3)
    plt.show()

    plot_pacf(values, lags=lags)
    fig = plt.gcf()
    fig.set_size_inches(16, 3)
    plt.show()

def plot_forecast(forecast, title='', save_path=None):
    sns.lineplot(x='visit_date', y='visitors', data=forecast, label='observed', alpha=0.9)
    sns.lineplot(x='visit_date', y='fitted', data=forecast, label='fitted', alpha=0.8)
    sns.lineplot(x='visit_date', y='forecast', data=forecast, label='forecast')
    ax = plt.gca()
    ax.fill_between(forecast['visit_date'],
                    forecast['95_lower'],
                    forecast['95_upper'],
                    color='k',
                    alpha=0.2)
    plot_config(title)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Number of Visitors')
    ax.set_xlabel('Date')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def plot_residuals(index, residuals, title=''):
    ax = plt.subplot2grid((1, 2), (0, 0))
    sns.distplot(residuals, ax=ax, norm_hist=True)
    sns.distplot(np.random.standard_normal(1000),
        kde=True, hist=False, norm_hist=True, label='N(0,1)')
    ax.set_title('Histogram plus estimated density')

    ax = plt.subplot2grid((1, 2), (0, 1))
    sm.qqplot(np.array(residuals), fit=True, line='45', ax=ax)
    fig = plt.gcf()
    fig.set_size_inches(16, 7)
    ax.set_title('Normal Q-Q')
    plt.show()

    plot_acf_pacf(index=index, values=residuals, lags=10, title='Standardized residual')

def plot_rolling_window_analysis(analysis, model, cutoff=0):
    residuals = analysis.loc[analysis['model'].str.contains(model), 'fitted_residual'].values[-1][cutoff:]
    plot_residuals(range(len(residuals)), residuals/np.std(residuals))
