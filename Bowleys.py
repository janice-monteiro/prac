import numpy as np
def bowley_skewness_coefficient(x): # calculate the median and quartiles
    q1= np.percentile (x, 25)
    q2= np.percentile (x, 50)
    q3= np.percentile(x, 75)
    # calculate Bowley's coefficient of skewness
    sk = (q1+q3 - 2*q2)/(q3-q1) 
    return sk

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
sk = bowley_skewness_coefficient(x)
print("Bowley's coefficient of skewness: ", sk)