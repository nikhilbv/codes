
import os

cmd = "git --version"

for x in range(0,3):
  returned_value = os.system(cmd)  # returns the exit code in unix
  print('returned value:', returned_value)

# import os;
# for x in range(0,3):
#     os.system("date")