import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from utils import *

output_dir = "outputs_apple"
os.makedirs(output_dir, exist_ok=True)

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

ts = pd.read_csv("apple_stock/res1.csv", sep=';', decimal=',', header=None)
ts = ts.iloc[:, :-21]

ref = ref['Close/Last'].tolist()
min_ref = min(ref)
ref = [x - min_ref for x in ref]
max_value = max(ref)
ref_norm = [x / max_value for x in ref]

# print(f"ref mean {sum(ref)/len(ref)}")


plot_time_series(*[normalize_series(ts.loc[i].tolist()) for i in range(10)],ref_serie=ref_norm, title=f'Apple Series Comparison', save_dir=output_dir)

# for i in range(len(ts)):

#     ts_i = ts.loc[i].tolist()
#     ts_i = [x - min(ts_i) for x in ts_i]
#     # print(f"ts {i} mean {sum(ts_i)/len(ts_i)}")
#     ts_i_norm = [x / max(ts_i) for x in ts_i]
#     print(f"\n=== Time Series {i} ===")
#     # print(f"DTW distance (standard): {compute_dtw_matrix(ref, ts_i)[-1][-1]}")
#     # print(f"DTW distance (restricted, window=20): {compute_dtw_matrix_restricted(ref, ts_i, 20)[-1][-1]}")

#     plot_time_series(ref_norm, ts_i_norm, title=f'{i} Apple Series Comparison', save_dir=output_dir)

#     plot_dtw_path(ref_norm, ts_i_norm, title=f'{i} DTW Matrix Standard', save_dir=output_dir)

#     plot_dtw_path_restricted(ref_norm, ts_i_norm, 20, title=f'{i} DTW Matrix Restricted', save_dir=output_dir)

#     plot_dtw_alignment(ref_norm, ts_i_norm, 'standard', title=f'{i} DTW Alignment Standard', save_dir=output_dir)

#     plot_dtw_alignment(ref_norm, ts_i_norm, 'restricted', window=20, title=f'{i} DTW Alignment Restricted', save_dir=output_dir)

