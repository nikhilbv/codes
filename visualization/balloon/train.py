import matplotlib.pyplot as plt
import numpy as np
# label = ['Balloon']
label = 'Balloon'
# no_balloon = [ 941 ]
no_balloon = 600

def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(label))
    # index = np.arange(len(label))
    plt.bar(1, no_balloon, width = 0.35)
    plt.xlabel('label', fontsize=10)
    plt.ylabel('No of balloons', fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.title('Train dataset of balloons')
    plt.show()

plot_bar_x()