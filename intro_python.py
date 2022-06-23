print("hello world")

#pandas
import pandas as pd
c = [1, 2,2]
d = pd.DataFrame(c)
print(d)

#random
import random
x =random.random()
print(x)

#string
y = "hello"
print(y[1:2])
print(y.replace("hello","this is fun"))

#list
list1 =[1,2,"hello","world"]
print(list1[1:])

list2 = [i for i in range(5,10)]
list3 = [i for i in range(10)]
print(list2)
list2.insert(1,"hello")
print(list2)
print(list2+list3)
#print reverse
print(list3[::-1])