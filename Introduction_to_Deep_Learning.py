#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import cProfile
import time


A = tf.constant([[1,2,3],[4,5,7],[5,7,3]])
tf.print(A)

# Sum of all elements in the matrix
sumOfA = tf.reduce_sum(A)
tf.print(sumOfA)


# Column wise addition [matrix_name, 0]
colAddOfA = tf.reduce_sum(A,0)
tf.print(colAddOfA) 


# Row wise addition [matrix_name, 1]
rowAddOfA = tf.reduce_sum(A,1)
tf.print(rowAddOfA)

B = tf.constant([1,2,3])
tf.print('B =',B)

C = tf.constant([4,5,6])
tf.print('C =',C)

# Element wise add 2 matrices 
sumBC = tf.add(B,C)
tf.print('sumBC =',sumBC)

# Element wise substract 2 matrices 
subBC = tf.subtract(B,C)
tf.print('subBC =',subBC)

# Element wise multiply 2 matrices 
mulBC = tf.multiply(B,C)
tf.print('mulBC =',mulBC)



D = tf.constant([[1,2],[8,9],[2,8]])
tf.print('D:\n',D)

E = tf.constant([[4],[3]])
tf.print('\nE:\n',E)

# Matrix multiplication 2 matrices 
matMulDE = tf.matmul(D,E)
tf.print('\nmatMulDE:\n',matMulDE)



D = tf.constant([[1,2,5,4],[8,9,1,9],[2,8,4,6]])
tf.print('D:\n',D)

E = tf.constant([[4],[3],[6],[5]])
tf.print('\nE:\n',E)

# Matrix multiplication 2 matrices 
matMulDE = tf.matmul(D,E)
tf.print('\nmatMulDE:\n',matMulDE)




# Now lets comapre the time comaprision between the ready matmul function & of a that we make for matrix multiplictaion.

def matmul_myFunc(X,Z):
    total = tf.matmul(X,Z)
    return total

myFuncAns = matmul_myFunc(D,E)
tf.print('\nmyFuncAns:\n',myFuncAns)



with cProfile.Profile() as pr:
    C3 = tf.matmul(D,E)
    pr.print_stats()
display(C3)



with cProfile.Profile() as pr:
    Y = matmul_myFunc(D,E)
    pr.print_stats()
display(Y)

