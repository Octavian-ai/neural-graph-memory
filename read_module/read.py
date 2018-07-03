#Input Data - Table (N-E-N)
import tensorflow as tf
import numpy as np
import pandas as pd

    
data=pd.read_csv('table.csv')
#convert the pandas object to a tensor
data=tf.convert_to_tensor(data)
print(data)

c = tf.placeholder(tf.float32, shape=(3))
m = tf.placeholder(tf.float32, shape=(3))
ti = tf.placeholder(tf.float32, shape=(3))
dot_c_m = tf.tensordot(c, m, 1)
dot_ti_m = tf.tensordot(ti, m, 1)
W = tf.eye(4)
result_array = np.array([])

with tf.Session() as sess:
    for i in range(5):
        result = dot_c_m.eval(feed_dict={c: [11, 0, 163], m: [1, 0, 1]})* W * (dot_ti_m.eval(feed_dict={ti: data[i].eval(), m: [1, 0, 1]}))
        result_array = np.append(result_array, result.eval())

print(result_array)   
        

