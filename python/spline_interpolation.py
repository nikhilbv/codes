import numpy as np

def getEquidistantPoints(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts+1),
               np.linspace(p1[1], p2[1], parts+1))


def _getEquidistantPoints(p1, p2, parts):
    return np.linspace(p1, p2, parts+1)


x = [409, 501, 551, 577, 585, 587, 586, 580, 571, 565, 551, 541, 536, 521, 513, 503, 491, 489, 473, 469, 449, 443, 435, 431, 426, 415, 403, 389, 399, 387, 377, 358, 367, 354, 356, 344, 318, 327, 336, 314, 317, 296, 305, 282, 285, 288, 261, 269, 277, 245, 249, 252]
y = [202, 211, 221, 233, 242, 253, 261, 271, 285, 291, 305, 316, 321, 336, 345, 356, 368, 370, 387, 391, 412, 419, 427, 435, 445, 456, 470, 484, 474, 490, 508, 529, 519, 542, 532, 558, 588, 578, 567, 600, 590, 627, 617, 659, 649, 639, 687, 677, 666, 721, 711, 701]

ip_x = []
ip_y = []
# ip = []

# pts = [(a,b) for a,b in zip(x,y)]
# for i in range((len(pts) -1)):
#   ip.append(list(getEquidistantPoints(pts[i], pts[i+1], 4)))
# print(ip)

for i in range((len(x) -1)):
  ip_x.append(list(_getEquidistantPoints(x[i], x[i+1], 4)))
  ip_y.append(list(_getEquidistantPoints(y[i], y[i+1], 4)))

flat_list = [int(item) for sublist in ip_x for item in sublist]
mylist = list(dict.fromkeys(flat_list))

print(mylist)

