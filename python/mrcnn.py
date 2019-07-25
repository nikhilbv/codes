import numpy as np
import math

h=1080
w=1920

# n1=1080
# n2=1920

ratio=np.arange(1.70,1.80,0.01)
# print(ratio)

x=[]
for n1 in range(64,1000):
  # print("n1:{}".format(n1))
  if not n1%64==0:
    continue
  for n2 in range(64,1000):
    # print("n2:{}".format(n2))
    if not n2%64==0:
      continue
    product=n2*n1
    # x.append(product)
    remainder=float(n2)/float(n1)
    # print("remainder:{}".format(remainder))
    # floor_of_r=math.floor(remainder)
    floor_of_r=round(remainder,2)
    # print("product%64==0 and floor_of_r in ratio: {}".format(product%64==0 and floor_of_r in ratio))
    if product%64==0 and floor_of_r in ratio:
      print(n1,n2, float(n2)/float(n1))

# [{'w': 320, 'h': 192, 'ap': 1.67}, {'w': 448, 'h': 256, 'ap': 1.75}, {'w': 576, 'h': 320, 'ap': 1.8}, {'w': 640, 'h': 384, 'ap': 1.67}, {'w': 704, 'h': 384, 'ap': 1.83}, {'w': 768, 'h': 448, 'ap': 1.71}, {'w': 896, 'h': 512, 'ap': 1.75}, {'w': 960, 'h': 576, 'ap': 1.67}, {'w': 1024, 'h': 576, 'ap': 1.78}]

# dims = [(2**6)*i for i in range(1,50) if (2**6)*i < 1080]
# [{'w':i,'h':dims[j],'ap':round(i/dims[j],2)} for j in range(0,len(dims)) for i in dims[j:] if 1.6 < round(i/dims[j],1) < 1.9 ]
# ({'w':i,'h':dims[j],'ap':round(i/dims[j],2)} for j in range(0,len(dims)) for i in dims[j:] if 1.6 < round(i/dims[j],1) < 1.9 )
