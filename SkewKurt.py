from scipy.stats import skew, kurtosis
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics

def calculate_skewness(x):
    sk = skew(x)
    return sk

def calculate_kurtosis(x):
    kurt = kurtosis(x)
    return kurt

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) #Calculating mean and standard deviation
mean = statistics.mean(x)
sd = statistics.stdev(x)
plt.plot(x, norm.pdf(x, mean, sd))
plt.show()

sk = calculate_skewness(x) 
kurt = calculate_kurtosis(x)
print("Skewness: ", sk)
print("Kurtosis: ", kurt)