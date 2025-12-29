print("Hello World !")
import copy
y = [1,2,3,[5,6]]

#shallow copy 

x = copy.copy(y)
x[3][0] = 10   
print(y)     # make a shallow copy of y  
x = copy.deepcopy(y)
x[3][0] = 10 
print(y)