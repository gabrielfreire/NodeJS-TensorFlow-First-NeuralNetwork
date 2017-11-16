import sys, json, numpy as np, tensorflow as tf

def read_in():
    lines = sys.stdin.readlines()
    return json.loads(lines[0])

lines = read_in()
np_lines = np.array(lines)

# #data
x_data = np.array(np_lines[0])
y_data = np.array(np_lines[1])


n_input = 2
n_hidden = 10
n_output = 1

W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0), dtype=tf.float32, name="W1")
W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0), dtype=tf.float32, name="W2")
b1 = tf.Variable(tf.zeros([n_hidden]), dtype=tf.float32, name="Bias1")
b2 = tf.Variable(tf.zeros([n_output]), dtype=tf.float32, name="Bias2")
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32,name="Y")

L2 = tf.sigmoid(tf.matmul(X,W1) + b1)
# prediction hypoteses
hy = tf.sigmoid(tf.matmul(L2,W2) + b2)

def main():
    #create matrices
    x = []
    y = []

    # populate the matrices
    for i in range(len(x_data)):
        x.append(x_data[i])
        y.append(y_data[i])
    
    #create the saver to load a model
    saver = tf.train.Saver()
    # create a Session
    with tf.Session() as sess:
        # We restore the saved trained model's weights
        saver.restore(sess, 'tmp/model.ckpt-9000')
        # run the NN calculating the hypoteses passing an Test Input
        y_out = sess.run([hy], feed_dict={X: x})
        #print the output
        to_nodejs = json.dumps(np.array(y_out[0]).tolist())
        print(to_nodejs)
        
if __name__ == '__main__':
    main()