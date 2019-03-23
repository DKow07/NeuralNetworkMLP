import numpy as np
import csv

def get_data(path):
    data = readcsv(path)
    d = np.empty((3437, 9), dtype=float)
    for i, obj in enumerate(data):
        for j, o in enumerate(obj):
            if j == 9:
                continue
            d[i, j] = float(obj[j+1]) / 10000

    return d


def get_labels(path):
    data = readcsv(path)
    d = np.empty((3437, 1), dtype=float)
    for i, obj in enumerate(data):
            d[i, 0] = float(obj[0])
    return d


def readcsv(filename):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=";")
    rownum = 0
    a = []
    for row in reader:
        a.append(row)
        rownum += 1

    ifile.close()
    return a

def data_to_float(data):

    print(len(data))
    print(len(data[0]))
    for i, str in enumerate(data):
        for j, obj in enumerate(data[0]):
            data[i][j] = float(data[i][j])

    return data

