import numpy as np
d7 = [
    [937.91, 89.28 , 87.95],
    [1017.24, 90.24 , 90.02],
    [838.53, 87.45 , 90.19],
    [895.11, 87.84 , 81.71],
    [1477.69, 89.89 , 553.97],
    [1025.65, 93.82 , 93.86],
    [1027.81, 94.84 , 95.2],
    [1020.45, 97.27 , 94.2],
    [1011.59, 94.83 , 95.76],
    [2023.43, 174.64, 194.48]
]

d8 = [
    [1026.87, 92.21, 94.86],
    [1011.58, 94.31, 93.6],
    [1025.11, 94.19, 104.42],
    [942.06 , 92.3 , 85.37],
    [1492.82, 94.92, 567.96],
    [1023.99, 92.83, 93.94],
    [1023.11, 93.4 , 93.19],
    [1043.57, 97.68, 93.2],
    [1028.68, 94.82, 96.43],
    [2136.21, 189.42 , 205.53]
]

d9 = [
    [1025.7, 88.94, 93.09],
    [1007.39, 96.93, 92.19],
    [1055.96, 93.98, 104.61],
    [905.99, 80.55, 82.48],
    [1467.47, 94.22, 554.3],
    [1015.7, 94.56, 92.27],
    [1003.48, 90.1 , 92.2],
    [1026.28, 94.45, 91.43],
    [1055.59, 95.67, 95.41],
    [2058.8, 181.95, 195.5]
]

d10 = [
    [230.67, 21.19, 30.05],
    [215.16, 20.37, 32.51],
    [225.59, 19.91, 37.37],
    [208.35, 18.72, 27.16],
    [633.51, 23.19, 455.91],
    [225.61, 20.38, 34.5],
    [221.62, 20.02, 31.2],
    [222.16, 21.74, 30.54],
    [212.47, 19.82, 31.89],
    [429.1, 40.57, 59.39]
]

dataset = d10
for item in dataset:
    print(round(item[0]/item[1], 2), round(item[0]/item[2], 2))


speedup_mpi = [
    [10.51, 11.14, 11.53 , 10.89],
    [11.27, 10.73, 10.39 , 10.56],
    [9.59 , 10.88, 11.24 , 11.33],
    [10.19, 10.21, 11.25 , 11.13],
    [16.44, 15.73, 15.57 , 27.32],
    [10.93, 11.03, 10.74 , 11.07],
    [10.84, 10.95, 11.14 , 11.07],
    [10.49, 10.68, 10.87 , 10.22],
    [10.67, 10.85, 11.03 , 10.72],
    [11.59, 11.28, 11.32 , 10.58]
]

speedup_mp = [
    [10.66, 10.83, 11.02, 7.68],
    [11.3, 10.81, 10.93, 6.62],
    [9.3 , 9.82, 10.09, 6.04],
    [10.95, 11.04, 10.98, 7.67],
    [2.67, 2.63 , 2.65 , 1.39],
    [10.93, 10.9, 11.01, 6.54],
    [10.8, 10.98, 10.88, 7.1],
    [10.83, 11.2, 11.22, 7.27],
    [10.56, 10.67, 11.06, 6.66],
    [10.4, 10.3, 10.53, 7.23]
]

print()
for item in speedup_mp:
    print(round(np.average(item), 2))

# Run:
# $ python test/speed-up.py