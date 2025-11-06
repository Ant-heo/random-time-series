import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, sys

from utils import *

original_stdout = sys.stdout

def save_distance_measures(output_dir, distances):
    sys.stdout = open(f"{output_dir}/measures.txt", "w", encoding="utf-8")
    print("DTW Distances:", distances)
    print("Average DTW Distance:", sum(distances) / len(distances))
    print("Minimum DTW Distance:", min(distances))
    print("Maximum DTW Distance:", max(distances))
    sys.stdout.close()
    sys.stdout = original_stdout

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
        dist = get_mean_distance(dtw_matrix)
        distances.append(dist)
        round_dist = round(dist, 1)
        round_dist_restricted = round(get_mean_distance(dtw_matrix_restricted), 1)
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
    save_distance_measures(output_dir, distances)

def compare_weather():
    output_dir = "outputs_weather"
    os.makedirs(output_dir, exist_ok=True)

    ts_weather = load_generated_time_series("weather_data/10series.csv")
    ref_weather = load_weather_reference_data()

    # print(type(ts_weather))
    distances = compare_series(ref_weather, ts_weather, output_dir, save_plots=True)
    save_distance_measures(output_dir, distances)


def compare_weather_downsampled(downsample_factor):
    output_dir = f"outputs_weather_downsampled_{downsample_factor}"
    os.makedirs(output_dir, exist_ok=True)

    ts_weather = load_generated_time_series("weather_data/10series.csv")
    ref_weather = load_weather_reference_data()

    ts_weather_downsampled = [ts[::downsample_factor] for ts in ts_weather]
    ref_weather_downsampled = ref_weather[::downsample_factor]
    print(f"Length of weather data: {len(ts_weather[0])}")
    print(f"Length of downsampled weather data: {len(ts_weather_downsampled[0])}")
    print(f"Length of reference weather data: {len(ref_weather)}")
    print(f"Length of downsampled reference weather data: {len(ref_weather_downsampled)}")

    # print(type(ts_weather))
    distances = compare_series(ref_weather_downsampled, ts_weather_downsampled, output_dir, save_plots=True)
    save_distance_measures(output_dir, distances)

def compare_apple_smooth():
    output_dir = f"outputs_apple_smooth"
    os.makedirs(output_dir, exist_ok=True)

    ts_apple = load_generated_time_series("apple_stock/res1.csv")
    ref_apple = load_apple_reference_data()

    ts_apple_smoothed = list(map(smooth_ts, ts_apple))
    ref_apple_smoothed = smooth_ts(ref_apple)

    distances = compare_series(ref_apple_smoothed, ts_apple_smoothed, output_dir, save_plots=True)
    save_distance_measures(output_dir, distances)

def compare_weather_smooth():
    output_dir = f"outputs_weather_smooth"
    os.makedirs(output_dir, exist_ok=True)

    ts_weather = load_generated_time_series("weather_data/10series.csv")
    ref_weather = load_weather_reference_data()
    downsample_factor = 30
    ts_weather_downsampled = [ts[::downsample_factor] for ts in ts_weather]
    ref_weather_downsampled = ref_weather[::downsample_factor]

    ts_weather_smoothed = [smooth_ts(ts_i, window_size=20) for ts_i in ts_weather_downsampled]
    ref_weather_smoothed = smooth_ts(ref_weather_downsampled, window_size=20)

    # print(type(ts_weather))
    distances = compare_series(ref_weather_smoothed, ts_weather_smoothed, output_dir, save_plots=True)
    save_distance_measures(output_dir, distances)


def calculate_mean_distance():
    weather_series = load_generated_time_series("weather_data/10series.csv")
    weather_series = [ts[::30] for ts in weather_series]
    print(len(weather_series[0]))
    weather_series = [normalize_series(ts_i) for ts_i in weather_series]
    
    apple_series = load_generated_time_series("apple_stock/res1.csv")
    apple_series = [normalize_series(ts_i) for ts_i in apple_series]
    dtw_distances = []

    plot_time_series(*(apple_series+weather_series), ref_serie=None, title=f'Apple Generated Series Comparison', save_dir=".")

    for i in range(len(apple_series)):
        dtw_distances.append([])
        for j in range(len(weather_series)):
            dtw_matrix = compute_dtw_matrix(apple_series[i], weather_series[j])
            dtw_distances[i].append(get_mean_distance(dtw_matrix))
            print(f"Computed DTW distance between apple series {i} and weather series {j}: {dtw_matrix[-1][-1]}")

    
    flatten_distances = np.array(dtw_distances).flatten()
    mean = np.mean(flatten_distances)
    max = np.max(flatten_distances)
    min = np.min(flatten_distances)
    
    sys.stdout = open(f"mixed_distance_measures.txt", "w", encoding="utf-8")
    print(f"Computed the distances between {len(apple_series)} apple series and {len(weather_series)} weather series.")
    print(f"DTW Distances: {dtw_distances}")
    print(f"Average DTW Distance: {mean}")
    print(f"Minimum DTW Distance: {min}")
    print(f"Maximum DTW Distance: {max}")
    return


if __name__ == "__main__":
    compare_apple_stock()
    # compare_weather()
    # compare_weather_downsampled(30)
    # compare_weather_downsampled(80)
    # compare_apple_smooth()
    # compare_weather_smooth()
    # calculate_mean_distance()
    pass