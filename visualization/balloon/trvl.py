import numpy as np
import matplotlib.pyplot as plt

n_groups = 1
no_train = 255
no_val = 50

# fig = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, no_train, bar_width,
alpha=opacity,
color='b',
label='Train')
 
rects2 = plt.bar(index + bar_width, no_val, bar_width,
alpha=opacity,
color='g',
label='Val')
 
plt.xlabel('Balloons')
plt.ylabel('No of balloons')
plt.title('Train vs val')
plt.xticks(index + bar_width, ('Balloon'))
plt.legend()
 
plt.tight_layout()
plt.show()
