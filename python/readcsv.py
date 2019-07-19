import sys
import csv
import pandas as pd
import numpy as np

filename = sys.argv[1]
f = open(filename, 'r')
fileData = f.read()
fileS = fileData.split("\n")
# print(fileS)
header = []
data = []
for i,row in enumerate(fileS):
  if i == 0:
    headerP = row.split(",")
    header=headerP[2:] 
    # print(header)
  else:
    dataP = row.split(",")
    data.append(dataP[2:])
    # print(data)
print(data[0])
# for i in data:
  # print(i)


# data=pd.read_csv(filename,skiprows=2,usecols=[3,4,5,6,7],
#         names=['one','two','three','four','five'])
# data.head()
# print(data)
# to_plot=list(data)
# to_plot=np.array(data)
# print(type(to_plot))
# print(to_plot)
# for x in np.nditer(to_plot):
#   print(x)
# for i in to_plot:
#   print(i)

# for var in enumerate(data):
  # print (var)
# k=0

# for i in data:
#   print(i)

# for i,row in data.iterrows():
#   for col in data:
#     print(row)
# col=['one','two','three','four','five']
# for i,row in data.iterrows():
#   for 
#   print(data.ix[i,"one"])
# print(data.ix[1][0])  
# f = open(filename, 'r')
# csv = csv.reader(f)
# data = list(csv)
# print(type(data))
# for row in data:
#   print(row)
