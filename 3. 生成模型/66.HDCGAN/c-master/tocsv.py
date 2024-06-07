# Script to generate CSV file and pickle from labels.
# Curtó and Zarza.
# c@decurto.tw z@dezarza.tw

import os
import glob
import pandas as ps
import json as c
import pickle

def to_csv():
    path_to = 'labels/'
    files = [position for position in os.listdir(path_to)]
    list_csv = []
    for z in files:
        file_data = open('labels/{}'.format(z))   
        data = c.load(file_data)
        value = (z + '.png', data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10])
        list_csv.append(value)

    name_column = ['Filename', 'Age', 'Ethnicity', 'Eyes Color', 'Facial Hair', 'Gender', 'Glasses', 'Hair Color', 'Hair Covered', 'Hair Style', 'Smile', 'Visible Forehead']
    df_csv = ps.DataFrame(list_csv, columns = name_column)
    labels_train = list(set(list_csv))
    with open("c&z.p", "wb") as fp:   #Pickling
        pickle.dump(labels_train, fp)
    return df_csv

def main():
        df_csv = to_csv()
        df_csv.to_csv('c&z.csv', index = None)
        print('Successfully converted to csv.')

main()
