import numpy as np
import urllib.request

# BINARY CLASSIFICATION using Breast Cancer data set from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer

# URL for the Breast Cancer Data Set (UCI Machine Learning Repository)
input_file = './static/breast-cancer.data'
output_file = './static/cleansed-breast-cancer.data'
data_set = []

# COLLECT THE DATA ==============================================================================
# read the data set into a list
with open(input_file, 'r') as f:
    file_contents = f.readlines()

# CLEANSE THE DATA ==============================================================================
# parse the data set
for line in file_contents:
    data_set.append(line.split(","))

# cleanse the data, can only work with numbers with the classifiers
# attribute information from: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer
# 1. Class: no-recurrence-events, recurrence-events
# 2. age: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99.
# 3. menopause: lt40, ge40, premeno.
# 4. tumor-size: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59.
# 5. inv-nodes: 0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26, 27-29, 30-32, 33-35, 36-39.
# 6. node-caps: yes, no.
# 7. deg-malig: 1, 2, 3.
# 8. breast: left, right.
# 9. breast-quad: left-up, left-low, right-up,	right-low, central.
# 10. irradiat:	yes, no.

# There is missing data in the data set, need to fill in missing values!
for idx, item in enumerate(data_set):

    print("before data cleansing data[{}]=[{}]".format(idx, data_set[idx]))

    # map class [no-recurrence-events, recurrence-events] -> [0,1]
    if item[0] == "recurrence-events":
        data_set[idx][0] = 1
    elif item[0] == "no-recurrence-events":
        data_set[idx][0] = 0
    elif item[0] == "?":
        data_set[idx][0] = 0

    # map age [10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99] -> [1,2,3,4,5,6,7,8,9]
    if item[1] == '10-19':
        data_set[idx][1] = 10
    elif item[1] == '20-29':
        data_set[idx][1] = 20
    elif item[1] == '30-39':
        data_set[idx][1] = 30
    elif item[1] == '40-49':
        data_set[idx][1] = 40
    elif item[1] == '?':
        data_set[idx][1] = 40
    elif item[1] == '50-59':
        data_set[idx][1] = 50
    elif item[1] == '60-69':
        data_set[idx][1] = 60
    elif item[1] == '70-79':
        data_set[idx][1] = 70
    elif item[1] == '80-89':
        data_set[idx][1] = 80
    elif item[1] == '90-99':
        data_set[idx][1] = 90

    # map menopause [lt40, ge40, premeno] -> [1,2,3]
    if item[2] == 'lt40':
        data_set[idx][2] = 1
    elif item[2] == 'ge40':
        data_set[idx][2] = 2
    elif item[2] == '?':
        data_set[idx][2] = 2
    elif item[2] == 'premeno':
        data_set[idx][2] = 3

    # map tumor-size [0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59] -> [0,5,10,15,20,25,30,35,40,45,50,55]
    if item[3] == '0-4':
        data_set[idx][3] = 0
    elif item[3] == '5-9':
        data_set[idx][3] = 5
    elif item[3] == '10-14':
        data_set[idx][3] = 10
    elif item[3] == '15-19':
        data_set[idx][3] = 15
    elif item[3] == '20-24':
        data_set[idx][3] = 20
    elif item[3] == '25-29':
        data_set[idx][3] = 25
    elif item[3] == '?':
        data_set[idx][3] = 25
    elif item[3] == '30-34':
        data_set[idx][3] = 30
    elif item[3] == '35-39':
        data_set[idx][3] = 35
    elif item[3] == '40-44':
        data_set[idx][3] = 40
    elif item[3] == '45-49':
        data_set[idx][3] = 45
    elif item[3] == '50-54':
        data_set[idx][3] = 50
    elif item[3] == '55-59':
        data_set[idx][3] = 55

    # map inv-nodes [0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26, 27-29, 30-32, 33-35, 36-39] -> [0,3,6,9,12,15,18,21,24,27,30,33,36]
    if item[4] == '0-2':
        data_set[idx][4] = 0
    elif item[4] == '3-5':
        data_set[idx][4] = 3
    elif item[4] == '6-8':
        data_set[idx][4] = 6
    elif item[4] == '9-11':
        data_set[idx][4] = 9
    elif item[4] == '?':
        data_set[idx][4] = 9
    elif item[4] == '12-14':
        data_set[idx][4] = 12
    elif item[4] == '15-17':
        data_set[idx][4] = 15
    elif item[4] == '18-20':
        data_set[idx][4] = 18
    elif item[4] == '21-23':
        data_set[idx][4] = 21
    elif item[4] == '24-26':
        data_set[idx][4] = 24
    elif item[4] == '27-29':
        data_set[idx][4] = 27
    elif item[4] == '30-32':
        data_set[idx][4] = 30
    elif item[4] == '33-35':
        data_set[idx][4] = 33
    elif item[4] == '36-39':
        data_set[idx][4] = 36

    # map node-caps [yes, no] -> [1,0]
    if item[5] == 'yes':
        data_set[idx][5] = 1
    elif item[5] == 'no':
        data_set[idx][5] = 0
    elif item[5] == '?':
        data_set[idx][5] = 0

    # map deg-malig ['1', '2', '3'] -> [1,2,3]; str to numeric
    if item[6] == '1':
        data_set[idx][6] = 1
    elif item[6] == '?':
        data_set[idx][6] = 2
    elif item[6] == '2':
        data_set[idx][6] = 2
    elif item[6] == '3':
        data_set[idx][6] = 3

    # map breast [left, right] -> [1,2]
    if item[7] == 'left':
        data_set[idx][7] = 1
    elif item[7] == 'right':
        data_set[idx][7] = 2
    elif item[7] == '?':
        data_set[idx][7] = 2

    # map breast-quad [left-up, left-low, right-up,	right-low, central] -> [1,2,3,4,5]
    if item[8] == 'left_up':
        data_set[idx][8] = 1
    elif item[8] == 'left_low':
        data_set[idx][8] = 2
    elif item[8] == 'right_up':
        data_set[idx][8] = 3
    elif item[8] == '?':
        data_set[idx][8] = 3
    elif item[8] == 'right_low':
        data_set[idx][8] = 4
    elif item[8] == 'central':
        data_set[idx][8] = 5

    # map irradiat [yes, no] -> [1,0]
    if item[9] == 'yes\n':
        data_set[idx][9] = 1
    elif item[9] == 'no\n':
        data_set[idx][9] = 0
    elif item[9] == '?\n':
        data_set[idx][9] = 0

    print("after data cleansing data[{}]=[{}]".format(idx, data_set[idx]))

# write the cleansed data set from a list into a file
with open(output_file, 'w') as f:
    for line in data_set:
        data_record = ''

        # need to create our own data record from the 2d array
        for field in line:
            data_record += str(field) + ","

        # get rid of last ',' char
        data_record = data_record[:-1]

        f.write('{}\n'.format(data_record))

# CLEANSE THE DATA ==============================================================================
# CLEANSE THE DATA ==============================================================================
# CLEANSE THE DATA ==============================================================================

# data_set_target = data_set[:, 0]

dataset = np.loadtxt(output_file, delimiter=",")

# separate the data from the target attributes

# load the CSV file as a numpy matrix
# dataset = np.loadtxt(raw_data, delimiter=",")
# print(dataset.shape)
#
# # separate the data from the target attributes
# X = dataset[:, 0:7]
# Y = dataset[:, 8]
#
# print("X={}".format(X))
# print("Y={}".format(Y))
