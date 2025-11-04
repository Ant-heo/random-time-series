import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from utils import *



def load_apple_reference_data():
    ref = pd.read_csv("apple_stock/HistoricalQuotes.csv")
    ref.columns = ref.columns.str.strip()
    ref['Date'] = pd.to_datetime(ref['Date'], format='%m/%d/%Y')
    ref['Close/Last'] = ref['Close/Last'].str.replace(r'[$\s]', '', regex=True).astype(float)
    ref = ref.sort_values(by='Date')
    max_date = ref['Date'].max()
    start_date = max_date - pd.DateOffset(years=1)
    ref = ref[(ref['Date'] >= start_date) & (ref['Date'] <= max_date)]
    return ref['Close/Last'].tolist()

def load_weather_reference_data():
    ref = pd.read_csv("weather_data/weather_madrid_2019-2022.csv")

    ref = ref.iloc[2450:8760+2450, 2].reset_index(drop=True)
    return ref.tolist()
    
    

def load_generated_time_series(path):
    ts = pd.read_csv(path, sep=';', decimal=',', header=None)
    # print(np.where(pd.isnull(ts))[1][0])
    metadata_cols = np.where(pd.isnull(ts))[1][0]
    ts = ts.iloc[:, :metadata_cols]
    # print(ts)
    return [ts.loc[i].tolist() for i in range(len(ts))]

def compare_series(ref_serie, ts_series, output_dir, save_plots=False):
    ref_norm = normalize_series(ref_serie)
    ts_norm = [normalize_series(ts_i) for ts_i in ts_series]
    
    # plot all time series together
    plot_time_series(*ts_norm, ref_serie=ref_norm, title=f'Series Comparison', save_dir=output_dir)
    window_size = len(ref_norm) // 10
    distances = []
    print(f"Comparing {len(ts_norm)} time series to reference series...")
    for i in range(len(ts_norm)):
        dtw_matrix = compute_dtw_matrix(ref_norm, ts_norm[i])
        dtw_matrix_restricted = compute_dtw_matrix_restricted(ref_norm, ts_norm[i], window_size)
        distances.append(dtw_matrix[-1][-1])
        round_dist = round(dtw_matrix[-1][-1], 1)
        round_dist_restricted = round(dtw_matrix_restricted[-1][-1], 1)
        # distances.append(compute_dtw_matrix_restricted(ref_norm, ts_norm[i], 100)[-1][-1])
        if save_plots:
            print(f"\nPlotting Time Series {i}")
            # print(f"plot_time_series {i}.")
            plot_time_series(ref_norm, ts_norm[i], title=f'{i} Series Comparison', save_dir=output_dir)
            # print(f"plot_dtw_path {i}.")
            plot_dtw_path(ref_norm, ts_norm[i], title=f'{i} DTW Matrix Standard, distance={round_dist}', save_dir=output_dir, dtw_matrix=dtw_matrix)
            plot_dtw_path_restricted(ref_norm, ts_norm[i], window_size=window_size, title=f'{i} DTW Matrix Restricted, w={window_size}, distance={round_dist_restricted}', save_dir=output_dir, dtw_matrix=dtw_matrix_restricted)
            # print(f"plot_dtw_alignment {i}.")
            plot_dtw_alignment(ref_norm, ts_norm[i], 'standard', title=f'{i} DTW Alignment Standard', save_dir=output_dir, dtw_matrix=dtw_matrix)
            plot_dtw_alignment(ref_norm, ts_norm[i], 'restricted', window=window_size, title=f'{i} DTW Alignment Restricted', save_dir=output_dir, dtw_matrix=dtw_matrix_restricted)
        # print(f"DTW Distance for series {i}: {distances[-1]}")
    return distances

def compare_apple_stock():
    output_dir = "outputs_apple"
    os.makedirs(output_dir, exist_ok=True)

    ts_apple = load_generated_time_series("apple_stock/res1.csv")
    print(type(ts_apple))
    ref_apple = load_apple_reference_data()

    distances = compare_series(ref_apple, ts_apple, output_dir, save_plots=True)
    print("DTW Distances:", distances)
    print("Average DTW Distance:", sum(distances) / len(distances))
    print("Minimum DTW Distance:", min(distances))
    print("Maximum DTW Distance:", max(distances))

def compare_weather():
    output_dir = "outputs_weather"
    os.makedirs(output_dir, exist_ok=True)

    ts_weather = load_generated_time_series("weather_data/temperatures.csv")
    ref_weather = load_weather_reference_data()

    # print(type(ts_weather))
    distances = compare_series(ref_weather, ts_weather, output_dir, save_plots=True)
    print("DTW Distances:", distances)
    print("Average DTW Distance:", sum(distances) / len(distances))
    print("Minimum DTW Distance:", min(distances))
    print("Maximum DTW Distance:", max(distances))


def compare_weather_downsampled(downsample_factor):
    output_dir = f"outputs_weather_downsampled_{downsample_factor}"
    os.makedirs(output_dir, exist_ok=True)

    ts_weather = load_generated_time_series("weather_data/temperatures.csv")
    ref_weather = load_weather_reference_data()

    ts_weather_downsampled = [ts[::downsample_factor] for ts in ts_weather]
    ref_weather_downsampled = ref_weather[::downsample_factor]
    print(f"Length of weather data: {len(ts_weather[0])}")
    print(f"Length of downsampled weather data: {len(ts_weather_downsampled[0])}")
    print(f"Length of reference weather data: {len(ref_weather)}")
    print(f"Length of downsampled reference weather data: {len(ref_weather_downsampled)}")

    # print(type(ts_weather))
    distances = compare_series(ref_weather_downsampled, ts_weather_downsampled, output_dir, save_plots=True)
    print("DTW Distances:", distances)
    print("Average DTW Distance:", sum(distances) / len(distances))
    print("Minimum DTW Distance:", min(distances))
    print("Maximum DTW Distance:", max(distances))
    
    
if __name__ == "__main__":
    compare_apple_stock()
    compare_weather()
    compare_weather_downsampled(30)
    compare_weather_downsampled(80)
    pass