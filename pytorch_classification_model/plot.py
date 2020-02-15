import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pylab
from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot

fp = open('log_1.txt', 'r')
train_iterations = []
train_loss = []
test_iterations = []
test_accuracy = []
for ln in fp:
    arr = re.findall(r'ion \b\d+\b,',ln)
    break
