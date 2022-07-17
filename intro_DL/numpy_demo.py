import numpy as np
import matplotlib.pyplot as plt


a = np.ones([2,1])
print(a)


#add
a = a+2
print(a)
b = [[1],[2]]


#elementwise multiplication
c = a*b
print(c)


#matrix multiplication
d = np.matmul(c,a.T)
print(d)


#add vertical axis 
e = d.sum(axis=0)


#add horizontal axis
f = d.sum(axis=1)



#checkerboad pattern first element equal to 0
checker  = np.zeros([4,4])
checker[1::2, ::2] = 1
checker[::2, 1::2] = 1
print(checker)
plt.imshow(checker)
plt.show()


#flatten
imgs = np.split(checker, 4, axis=0)
img = np.hstack(imgs)
#plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.imshow(img)
plt.show()



#checkerboad pattern first element equal to 1
checker2  = np.zeros([4,4])
checker2[::2, ::2] = 1
checker2[1::2, 1::2] = 1
print(checker2)
