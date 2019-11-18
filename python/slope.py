x = [
      821,
      847,
      868,
      898,
      927,
      943,
      964,
      996,
      1007,
      1047,
      1071,
      1101,
      1105,
      1139,
      1178,
      1191,
      1206,
      1223
    ]
y = [
      251,
      261,
      271,
      283,
      294,
      302,
      311,
      323,
      329,
      345,
      355,
      366,
      368,
      383,
      401,
      406,
      411,
      418
    ]
print("x : {}".format(x))
print("x_type : {}".format(type(x)))
print("y : {}".format(y))
from scipy.stats import linregress
print(linregress(x, y))
