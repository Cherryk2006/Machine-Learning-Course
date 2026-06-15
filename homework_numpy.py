import numpy as np
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "target"
]

import requests
from io import StringIO

response = requests.get(url, verify=False)
response.raise_for_status()
data_text = StringIO(response.text)
df = pd.read_csv(data_text, header=None, names=columns)
df = df.dropna()

target = df["target"]
print("Target column:")
print(target.head())

X = df.drop("target", axis=1)
data = X.to_numpy()
print("Shape of array:", data.shape)

first_column = data[:, 0]
mean_value = np.mean(first_column)
median_value = np.median(first_column)
std_value = np.std(first_column)

print("\nStatistics for first column:")
print("Mean:", mean_value)
print("Median:", median_value)
print("Standard deviation:", std_value)

rows, cols = data.shape
total_positions = rows * cols
random_positions = np.random.choice(
    total_positions,
    size=20,
    replace=False
)
data_with_nan = data.astype(float).copy()
for pos in random_positions:
    row = pos // cols
    col = pos % cols
    data_with_nan[row, col] = np.nan

nan_positions_first_col = np.where(np.isnan(data_with_nan[:, 0]))[0]
print("\nPositions of NaN in first column:")
print(nan_positions_first_col)

data_with_nan = np.nan_to_num(data_with_nan, nan=0)
filtered_data = data_with_nan[
    (data_with_nan[:, 2] > 1.5) &
    (data_with_nan[:, 0] < 5.0)
]
print("\nFiltered array:")
print(filtered_data)

left_part, right_part = np.hsplit(
    data_with_nan,
    data_with_nan.shape[1] // 2
)

left_part = left_part[np.argsort(left_part[:, 0])]
right_part = right_part[np.argsort(right_part[:, 0])[::-1]]
combined_array = np.hstack((left_part, right_part))

unique_values, counts = np.unique(
    combined_array,
    return_counts=True
)
print("\nUnique values and counts:")
for value, count in zip(unique_values, counts):
    print(value, "->", count)

most_frequent_value = unique_values[np.argmax(counts)]
print("\nMost frequent value:")
print(most_frequent_value)

def transform_column(column):
    mean_col = np.mean(column)
    result = np.where(
        column < mean_col,
        column * 2,
        column / 4
    )
    return result

combined_array[:, 2] = transform_column(
    combined_array[:, 2]
)
print("\nThird column after transformation:")
print(combined_array[:, 2])

correlation_matrix = X.corr()
print("\nExtra task:")
print("Correlation matrix:")
print(correlation_matrix)

# Я вирішила додатково дослідити взаємозв’язок між ознаками датасету за допомогою кореляційної матриці.
# Це допомагає краще зрозуміти структуру даних та побачити, які характеристики найбільше пов’язані між собою.
# З аналізу видно, що довжина та ширина пелюстки мають сильну позитивну кореляцію.