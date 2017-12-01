import sys, json, numpy as np, tensorflow as tf
import os

dir = os.path.dirname(os.path.realpath(__file__))
#hyperparameters
n_input = 2
n_hidden = 10
n_output = 1

learning_rate = 0.01
epochs = 10000

# create placeholders for input and output of type float32 (all the variables must be of the same type)
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, name="X")

# create the Variables of type float32 and give them names
with tf.variable_scope('input_layer'):
    W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0), dtype=tf.float32, name="W1")
    b1 = tf.Variable(tf.zeros([n_hidden]), dtype=tf.float32, name="Bias1")
    input_layer = tf.sigmoid(tf.matmul(X, W1) + b1)
with tf.variable_scope('output_layer'):
    W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0), dtype=tf.float32, name="W2")
    b2 = tf.Variable(tf.zeros([n_output]), dtype=tf.float32, name="Bias2")
    prediction_output = tf.sigmoid(tf.matmul(input_layer, W2) + b2)

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, name="Y")
    global_step_t = tf.Variable(0, name="global_step", trainable=False)
    # cross entropy cost function faster
    cost = tf.reduce_mean(-Y * tf.log(prediction_output) - (1 - Y) * tf.log(1 - prediction_output), name="accuracy")

# Create a Gradient Descent optimizer
with tf.variable_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step_t, name="train_op")

with tf.variable_scope('logs'):
    tf.summary.scalar('cost_f', cost)
    summaries = tf.summary.merge_all()


all_saver = tf.train.Saver()

def read_in():
    lines = sys.stdin.readlines()
    # print(lines)
    return json.loads(lines[0])

lines = read_in()
np_lines = np.array(lines)

# # #data
x_data = np.array(np_lines[0])
y_data = np.array(np_lines[1])
#create matrices
# x = [
#         [0, 0],
#         [0, 1],
#         [1, 0],
#         [1, 1]
#     ]
# y = [
#         [0],
#         [0],
#         [1],
#         [1]
#     ]

x = []
y = []
def main():
    # populate the matrices
    for i in range(len(x_data)):
        x.append(x_data[i])
        y.append(y_data[i])

    # create a Session an initialize
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        #initialize the session
        session.run(init)
        # event log for tensorboard
        sw = tf.summary.FileWriter('events/', session.graph)
        # for each iteration
        for step in range(epochs):
            # X: input and Y: output
            feed_dict = {
                X: x,
                Y: y
            }
            
            # train the model using GradientDescent optimizer
            # session.run(optimizer, feed_dict=feed_dict)
            # what values to store in the event file
            to_compute = [optimizer, cost, global_step_t, summaries]
            # save/unpack those values in another array that looks exactly the same 
            train_op, cost_f, global_step, summaries_metric = session.run(to_compute, feed_dict=feed_dict)
            if step % 1000 == 0:
                # print the actual cost using the cost function and passing the input and output
                print(session.run(cost, feed_dict=feed_dict))
                #add a summary passing the metrics and global step
                sw.add_summary(summaries_metric, global_step)
                # save the model in this step (step % 1000 == 0)
                save_path = all_saver.save(session, "tmp/model.ckpt", global_step = step)
                print("Model saved in file: %s" % save_path)

        # calculate the error
        correct = tf.equal(tf.floor(prediction_output + 0.5), Y)
        # use the correct to calculate the accuracy
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))

        # print the prediction
        print(session.run([prediction_output], feed_dict=feed_dict))
        # print the accuracy
        print("Accuracy: ", accuracy.eval(feed_dict) * 100, "%")

if __name__ == '__main__':
    main()