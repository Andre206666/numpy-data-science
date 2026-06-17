import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print("\nArray:", arr)
print("Type:", type(arr))
print("Mean:", arr.mean())
print("Sum:", arr.sum())
print("Max:", arr.max())
print("Min:", arr.min())
print("Shape:", arr.shape)
print("Doubled:", arr * 2)

arr2 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print("\nMean:", arr2.mean())
print("Sum:", arr2.sum())
print("Min:", arr2.min())
print("Doubled:", arr2 * 0.1 / 10)
print("Filter: ", arr2[arr2 > 50])
print("Shape:", arr2.shape)

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix)
print("\nShape:", matrix.shape)
print("Mean:", matrix.mean())
print("Row means:", matrix.mean(axis=1))
print("Column means:", matrix.mean(axis=0))

matrix2 = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 11]
])
print(matrix2)

print("\nShape:", matrix2.shape)
print("Row means:", matrix2.mean(axis=1))
print("Column means:", matrix2.mean(axis=0))
print("Max per column:", matrix2.max(axis=0))


import numpy as np
import pandas as pd

data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

df = pd.DataFrame(data, columns=["A", "B", "C"])
print(df)
print(df.mean())


grades = np.array([
    [8, 7, 9],
    [6, 8, 7],
    [9, 9, 8],
    [5, 6, 7],
    [7, 8, 6],
])

df = pd.DataFrame(grades, columns=["Math", "Science", "English"])

print(df.mean())

print(df.mean(axis=1))

df["average"] = df.mean(axis=1)
print(df[df["average"] > 7])