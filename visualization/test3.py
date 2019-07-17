# library
import matplotlib.pyplot as plt
 
# Make fake dataset
height = [3, 12, 5, 18, 45]
bars = ('A', 'B', 'C', 'D', 'E')
# Choose the width of each bar and their positions
# width = [0.1,0.2,3,1.5,0.3]
y_pos = [0,1,2,3,4]
 
# Make the plot
# plt.bar(y_pos, height, width=width)
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.show()
