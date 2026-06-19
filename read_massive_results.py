import pandas as pd

try:
    df = pd.read_csv('D:/NIt/MAIR_Project/results/massive_benchmark_results.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('D:/NIt/MAIR_Project/results/massive_benchmark_results.csv', encoding='utf-16')
    except Exception as e:
        df = pd.read_csv('D:/NIt/MAIR_Project/results/massive_benchmark_results.csv', encoding='latin1')

print(df.to_string())
