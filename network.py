import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.biases = [tf.Variable(tf.random_normal([y])) for y in layers[1:]]
        self.weights = [tf.Variable(tf.random_normal([x, y])) for x, y in zip(layers[:-1], layers[1:])]
        self.x = tf.placeholder('float', shape=[None, layers[0]])
        self.y = tf.placeholder('float')
        self.saver = tf.train.Saver()

    def feed_forward(self, input_a):
        output_a = input_a
        for w, b in zip(self.weights, self.biases):
            output_a = tf.add(tf.matmul(output_a, w), b)
            if w == self.weights[-1]:
                return output_a
            output_a = tf.nn.relu(output_a)

    def train(self, epochs, batch_size, eta=None):
        prediction = self.feed_forward(self.x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        if eta:
            optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(cost)
        else:
            optimizer = tf.train.AdamOptimizer().minimize(cost)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                epoch_loss = 0
                for _ in range(int(mnist.train.num_examples / batch_size)):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})
                    epoch_loss += c
                print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Epoch', epoch, 'Accuracy:', accuracy.eval({self.x: mnist.test.images, self.y: mnist.test.labels}))
            save_path = self.saver.save(sess, 'resource/model.ckpt')
            print("Model saved in file: %s" % save_path)

    def predict(self, input):
        prediction = self.feed_forward(input)
        predicted_index = tf.argmax(prediction, 1)

        with tf.Session() as sess:
            self.saver.restore(sess, 'resource/model.ckpt')
            print sess.run(predicted_index)

def main():
    img = cv2.imread('testImages/mnist_6.jpg', 0)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
    img = img.reshape([1, 784])
    tensorInput = tf.convert_to_tensor(img,dtype=tf.float32)
    '''cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    net = Network([784,100,10])
    #net.train(50,100)
    net.predict(tensorInput)

if __name__ == "__main__":
    main()