import os
import sys
import argparse

import vfmodel as vfm
import vfplot as vfp

def main(args):
    parser = argparse.ArgumentParser(description='Make visitor forecasts.')
    parser.add_argument('store_id', metavar='store', type=str,
                        help='a string of the store id')
    parser.add_argument('forecast_horizon', metavar='n', type=int,
                        help='an integer for the number days to forecast')
    args = parser.parse_args(args)

    final, audit = vfm.forecast_for_store(args.store_id, args.forecast_horizon)

    final.to_csv('forecasts/reports/{}_{}days.csv'.format(audit['store_id'], audit['forecast_horizon']))
    vfp.plot_forecast(final,
                      title='Forecast for store {} with {} model and 95% Confidence Intervals'.format(
                        audit['store_id'], audit['best_model'].upper()),
                      save_path='forecasts/figures/{}_{}days'.format(audit['store_id'], audit['forecast_horizon']))

if __name__ == '__main__':
    main(sys.argv[1:])
