#Input Data - Table (N-E-N)
import tensorflow as tf
import numpy as np
import pandas as pd
#from sklearn.utils.extmath import softmax

#x = np.random.sample((100,2))
## make a dataset from a numpy array
#dataset = tf.data.Dataset.from_tensor_slices(x)
## create the iterator
#iter = dataset.make_one_shot_iterator()
#
#el = iter.get_next()
#
#with tf.Session() as sess:
#    print(sess.run(el)) # output: [ 0.42116176  0.40666069]

#ident_matrix_ex_1 = tf.eye(5) 
#print(ident_matrix_ex_1)
#
#with tf.Session() as sess:
#    print(sess.run(ident_matrix_ex_1))
    
data=pd.read_csv('table.csv')
#convert the pandas object to a tensor
data=tf.convert_to_tensor(data)
print(data)

#with tf.Session() as sess:
#    print(sess.run(data)) # output: [ 0.42116176  0.40666069]
#    
#def comparison_function(c,m,ti):
#    return 

c = tf.placeholder(tf.float32, shape=(3))
m = tf.placeholder(tf.float32, shape=(3))
ti = tf.placeholder(tf.float32, shape=(3))
s = tf.placeholder(tf.float32, shape=(1))
mul_c_m = tf.multiply(c, m)
mul_ti_m = tf.multiply(ti, m)
mul_ti_s = tf.multiply(ti, s)

W = tf.eye(3)

result_array = np.array([])
result_d = np.array([])

with tf.Session() as sess:
    for i in range(5):
        #result = dot_c_m(feed_dict={c: [11, 0, 163], m: [1, 0, 1]})#* W * (dot_ti_m.eval(feed_dict={ti: data[i].eval(), m: [1, 0, 1]}))
        #result_array = np.append(result_array, result.eval())
        #print(result.eval())
        result = (tf.tensordot((sess.run(mul_c_m, feed_dict={c: [49, 0, 197], m: [1, 0, 1]})), (tf.tensordot(W, (sess.run(mul_ti_m, feed_dict={ti: data[i].eval(), m: [1, 0, 1]})), axes=1).eval()), axes=1).eval())
        result_array = np.append(result_array, result)
    result_softmax = sess.run(tf.nn.softmax(result_array))
    for i in range(5):
        results = sess.run(mul_ti_s, feed_dict={ti: data[i].eval(), s: result_softmax[i]})
        result_d = np.append(result_d, results)
    
print(result_array)  
print(result_softmax[1]) 
print(results)
        

