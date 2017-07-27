import tensorflow as tf
import numpy as np

# Create 100 phone x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32) # np.random.rand() is a numpy library function which creates random numbers sampled uniformly [0, 1), where zero is possible and one is not possible.  In the paranthesis is the matrix size, which is in this case a one hundred length vector.  The astype changes how the number is stored/classified by the computer. 32 is known as single precision and 64 is known as double precision.
y_data = x_data * 0.1 + 0.3 # This is a line with slope 0.1 and intercept 0.3.
with tf.name_scope('x'):# for the graph
  x_data_beta = tf.convert_to_tensor(x_data, name="x_data") 
with tf.name_scope('y'):# for the graph
  y_data_beta = tf.convert_to_tensor(y_data, name="y_data")
# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
with tf.name_scope('W'): # for naming in the graph in tensorboard
  W = tf.Variable(tf.random_uniform([], -1.0, 1.0), dtype=tf.float32) # tf is the tensorflow library.  According to the tensorflow website, when training a model, variables hold and update parameters.  Variables contain tensors.  The [1] is the dimension,-1.0, 1.0 is the range for the uniform probability density.
with tf.name_scope('b'):
  b = tf.Variable(tf.zeros([]), dtype=tf.float32) # zeros creates a zero with the dimension in brackets.
y = W * x_data_beta + b # Note that y is now a function of two variables and a numpy array of random numbers.

#This will be used to make summaries in the loop below.
#W_placeholder= tf.placeholder(tf.float32)
#b_placeholder= tf.placeholder(tf.float32) #tf.float32
# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data_beta))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Add scalar summaries
#with tf.name_scope('W'):
tf.summary.scalar('W',W)
tf.summary.scalar('loss',loss)
#with tf.name_scope('b'):
tf.summary.scalar('b',b)

# Before starting, initialize the variables.  We will 'run' this first.
# init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Merge the summaries and write them.
merge = tf.summary.merge_all()
introduction_writer = tf.summary.FileWriter('~/Desktop/', sess.graph)

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
#        W_placeholder = sess.run(W)
#        b_placeholder = sess.run(b)
        summary = sess.run(merge)
#        W_present = sess.run(W)
#        b_present = sess.run(b)
#        _, W_summary_gamma = sess.run([train, merge], feed_dict={W_placeholder: W_present, b_placeholder: b_present})
        introduction_writer.add_summary(summary, step)      

introduction_writer.close()
# Learns best fit is W: [0.1] b: [0.3]
