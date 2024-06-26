

***Design Linear and Non Linear Perceptron Model***
"""

import numpy as np
import tensorflow as tf

def stepFunction(x):
  if x>=0:
    return 1
  else:
    return 0

#def step(x):
#    return tf.where(tf.math.greater_equal(x, 0), 1.0, 0.0)

def perceptron(x,w,b):
 ans=np.matmul(w,x)+b
 return stepFunction(ans)

#def an_d(a, b, w):
    # Reshape a to have shape (1, 2)
 #   a = tf.reshape(a, (1, -1))
    # Reshape w to have shape (2, 1)
  #  w = tf.reshape(w, (-1, 1))
   # ans = tf.matmul(a, w) + b
   # return step(ans)

test1 = np.array([0,0])
test2 = np.array([0,1])
test3 = np.array([1,0])
test4 = np.array([1,1])

#test1 = tf.constant([0.0, 0.0])
#test2 = tf.constant([0.0,1.0])
#test3 = tf.constant([1.0,0.0])
#test4 = tf.constant([1.0,1.0])

"""***Neural Network for AND Function***


"""

def AND_logicFunction(x):
  w=np.array([1,1])
  b=-1.5
  return perceptron(x,w,b)

# def AND_logicFunction(x):
#   w=tf.Variable([1.0,1.0],  dtype=tf.float32)
#   b=tf.Variable([-1.5],  dtype=tf.float32)
#   return perceptron(x,w,b)

y1=AND_logicFunction(test1)
y2=AND_logicFunction(test2)
y3=AND_logicFunction(test3)
y4=AND_logicFunction(test4)

print("AND (0,0) := ", y1)
print("AND (0,1) := ", y2)
print("AND (1,0) := ", y3)
print("AND (1,1) := ", y4)

"""***Neural Network for OR Function***"""

def OR_logicFunction(x):
  w=np.array([1,1])
  b=-0.5
  return perceptron(x,w,b)

y1=OR_logicFunction(test1)
y2=OR_logicFunction(test2)
y3=OR_logicFunction(test3)
y4=OR_logicFunction(test4)

print("OR (0,0) := ", y1)
print("OR (0,1) := ", y2)
print("OR (1,0) := ", y3)
print("OR (1,1) := ", y4)

"""***Neural Network for NAND Function***"""

def NAND_logicFunction(x):
  w=np.array([-1,-1])
  b=1.5
  return perceptron(x,w,b)

y1=NAND_logicFunction(test1)
y2=NAND_logicFunction(test2)
y3=NAND_logicFunction(test3)
y4=NAND_logicFunction(test4)

print("NAND (0,0) := ", y1)
print("NAND (0,1) := ", y2)
print("NAND (1,0) := ", y3)
print("NAND (1,1) := ", y4)

"""***Neural Network for XOR Function***"""

def EXOR_logicFunction(a):
  one=OR_logicFunction(a)
  two=NAND_logicFunction(a)
  three=np.array([one,two])
  return AND_logicFunction(three)

y1=EXOR_logicFunction(test1)
y2=EXOR_logicFunction(test2)
y3=EXOR_logicFunction(test3)
y4=EXOR_logicFunction(test4)

print("EXOR (0,0) := ", y1)
print("EXOR (0,1) := ", y2)
print("EXOR (1,0) := ", y3)
print("EXOR (1,1) := ", y4)

"""***Using Switch Case***"""

x=int(input("Number_1: "))
y=int(input("Number_2: "))
ANS=np.array([x,y])

print("\nEnter\n1=>AND\n2=>NAND\n3=>OR\n4=>EXOR\n")
c=input("=> ")

match c:
    case "1":
         print("\nAND")
         print(ANS)
         print(AND_logicFunction(ANS))
    case "2":
         print("\nNAND")
         print(ANS)
         print(NAND_logicFunction(ANS))
    case "3":
         print("\nOR")
         print(ANS)
         print(OR_logicFunction(ANS))
    case "4":
         print("\nEXOR")
         print(ANS)
         print(EXOR_logicFunction(ANS))
    case _:
        print("\nWrong Input ")

!sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-plain-generic

! jupyter nbconvert --to pdf /content/drive/MyDrive/DL_Lab_2.ipynb

import tensorflow as tf

def step(x):
    return tf.where(tf.math.greater_equal(x, 0), 1.0, 0.0)

def an_d(a, b, w):
    # Reshape a to have shape (1, 2)
    a = tf.reshape(a, (1, -1))
    # Reshape w to have shape (2, 1)
    w = tf.reshape(w, (-1, 1))
    ans = tf.matmul(a, w) + b
    return step(ans)

a = tf.constant([1.0, 0.0])
w = tf.constant([1.0, 1.0])
b = -1.5

print(an_d(a, b, w))



