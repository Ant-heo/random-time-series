import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from utils import *

output_dir = "outputs_apple"
os.makedirs(output_dir, exist_ok=True)

def load_apple_reference_data():
    ref = pd.read_csv("apple_stock/HistoricalQuotes.csv")
    ref.columns = ref.columns.str.strip()
    ref['Date'] = pd.to_datetime(ref['Date'], format='%m/%d/%Y')
    ref['Close/Last'] = ref['Close/Last'].str.replace(r'[$\s]', '', regex=True).astype(float)
    ref = ref.drop('Volume', axis=1)
    ref = ref.drop('Open', axis=1)
    ref = ref.drop('High', axis=1)
    ref = ref.drop('Low', axis=1)
    ref = ref.sort_values(by='Date')
    max_date = ref['Date'].max()
    start_date = max_date - pd.DateOffset(years=1)
    ref = ref[(ref['Date'] >= start_date) & (ref['Date'] <= max_date)]
    # print(ref)
    ref = ref.drop('Date', axis=1)
    return ref['Close/Last'].tolist()

def load_generated_time_series(path):
    ts = pd.read_csv(path, sep=';', decimal=',', header=None)
    ts = ts.iloc[:, :-21]
    return [ts.loc[i].tolist() for i in range(len(ts))]

def compare_series(ref_serie, ts_series, output_dir, save_plots=False):
    ref_norm = normalize_series(ref_series)
    ts_norm = [normalize_series(ts_i) for ts_i in ts_series]
    
    # plot all time series together
    plot_time_series(*ts_norm,ref_serie=ref_norm, title=f'Apple Series Comparison', save_dir=output_dir)

    distances = []
    for i in range(len(ts_norm)):
        distances.append(compute_dtw_matrix(ref_norm, ts_norm[i])[-1][-1])
        if save_plots:
            print(f"\nPlotting Time Series {i}")
            plot_time_series(ref_norm, ts_norm[i], title=f'{i} Apple Series Comparison', save_dir=output_dir)
            plot_dtw_path(ref_norm, ts_norm[i], title=f'{i} DTW Matrix Standard', save_dir=output_dir)
            plot_dtw_path_restricted(ref_norm, ts_norm[i], 20, title=f'{i} DTW Matrix Restricted', save_dir=output_dir)
            plot_dtw_alignment(ref_norm, ts_norm[i], 'standard', title=f'{i} DTW Alignment Standard', save_dir=output_dir)
            plot_dtw_alignment(ref_norm, ts_norm[i], 'restricted', window=20, title=f'{i} DTW Alignment Restricted', save_dir=output_dir)
    
    return distances

ts = load_generated_time_series("apple_stock/res1.csv")
ref = load_apple_reference_data()

distances = compare_series(ref, ts, output_dir)
print("DTW Distances:", distances)
print("Average DTW Distance:", sum(distances) / len(distances))
print("Minimum DTW Distance:", min(distances))
print("Maximum DTW Distance:", max(distances))
