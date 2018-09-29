import numpy as np 

emt = np.array([np.newaxis,np.newaxis])
a = np.array([[1,2],[3,4]])
c = np.vstack((emt,a))
print(c)
b = np.array([5,6])
c = np.column_stack((a,b))
print(c)
