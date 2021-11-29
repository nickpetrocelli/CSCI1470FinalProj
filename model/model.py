import numpy as np
import tensorflow as tf
from get_data import get_examples
import math


class ConvToLinear(tf.keras.Model):
    def __init__(self, image_dim):
        super(ConvToLinear, self).__init__()
        self.image_dim = image_dim # should be 100

        # idea: create set of sequentials
        # to compress common operations

        # architecture is currently based on my architecture for hw2
        # seemed to work pretty well then

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.conv_1 = tf.keras.Sequential()

        self.conv_1.add(tf.keras.layers.Conv2D(filters=20, kernel_size=5, strides=2, padding='SAME'))
        self.conv_1.add(tf.keras.layers.BatchNormalization())
        self.conv_1.add(tf.keras.layers.ReLU())
        self.conv_1.add(tf.keras.layers.MaxPool2D(padding='SAME'))
         

        self.conv_2 = tf.keras.Sequential()

        self.conv_2.add(tf.keras.layers.Conv2D(filters=40, kernel_size=5, strides=1, padding='SAME'))
        self.conv_2.add(tf.keras.layers.BatchNormalization())
        self.conv_2.add(tf.keras.layers.ReLU())
        self.conv_2.add(tf.keras.layers.MaxPool2D(padding='SAME'))

        self.conv_3 = tf.keras.Sequential()

        self.conv_3.add(tf.keras.layers.Conv2D(filters=50, kernel_size=2, strides=1, padding='SAME'))
        self.conv_3.add(tf.keras.layers.BatchNormalization())
        self.conv_3.add(tf.keras.layers.ReLU())
        self.conv_3.add(tf.keras.layers.MaxPool2D(padding='SAME'))


        #linears 1
        self.lin_1 = tf.keras.layers.Dense((image_dim * image_dim) / 2, activation='relu')

        #final linear layer with input image_dim x image_dim
        # no activation, these are pure logits
        self.logit_layer = tf.keras.layers.Dense(image_dim * image_dim)

    def call(self, inputs_batch):
        """
        inputs_batch: batch_size x 100 x 100 x 1
        """ 
        inputs = tf.cast(inputs_batch, tf.float32)
        conv_output_1 = self.conv_1(inputs)

        conv_output_2 = self.conv_2(conv_output_1)

        conv_output_3 = self.conv_3(conv_output_2)

        # reshape for linear
        # taken from hw2
        lin_in = tf.reshape(conv_output_3, [conv_output_3.shape[0], -1])

        lin_1_out = self.lin_1(lin_in)

        outputs = self.logit_layer(lin_1_out)

        return outputs

    def loss(self, logit_batch, labels_batch):
        """
        output_batch: BATCH_SIZE x 10,000
        labels_batch: BATCH_SIZE x 100 x 100
        """
        labels = tf.convert_to_tensor(labels_batch, dtype=tf.float64)
        labels_flat = tf.reshape(labels, (labels.shape[0], self.image_dim * self.image_dim))

        #idea: sum of crossentropy loss
        # partially inspired by hw5, this is somewhat of a "reconstruction loss" of the output river network
        # though this is a classification problem, not a generation one.
        loss_calc = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, 
            reduction=tf.keras.losses.Reduction.SUM,
        )
        loss = loss_calc(labels_flat, logit_batch)
        return loss


def train(model, train_images, train_labels, batch_size = 100):


    # shuffle the input as I have in the homework
    shuffled_indices = tf.random.shuffle([i for i in range(len(train_images))])
    shuffled_inputs = tf.gather(train_images, shuffled_indices)
    shuffled_labels = tf.gather(train_labels, shuffled_indices)

    batch_index = 0
    while batch_index < len(shuffled_inputs):
        input_batch = shuffled_inputs[batch_index:batch_index+batch_size]
        label_batch = shuffled_labels[batch_index:batch_index+batch_size]

        with tf.GradientTape() as tape:
            logits = model.call(input_batch)
            loss = model.loss(logits, label_batch)
            print(f"{batch_index} / {len(train_images)} LOSS : {loss}")

        # from lab
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        batch_index = batch_index + batch_size



    return

def test(model, test_images, test_labels, batch_size = 100):
    # very similar to homework procedure
    
    # TODO need a better accuracy metric
    # currently just return average test loss

    losses = []
    batch_index = 0
    while batch_index < len(test_images):
        input_batch = test_images[batch_index:batch_index+batch_size]
        label_batch = test_labels[batch_index:batch_index+batch_size]

        logits = model.call(input_batch)
        loss = model.loss(logits, label_batch)
        losses.append(loss)
        print(loss)
        batch_index = batch_index + batch_size

    return tf.reduce_mean(tf.convert_to_tensor(losses))


def main():
    img_dirs = ['../data/imgs/network_1_50m/stream_network_1_buff_50m/']
    label_dirs = ['../data/imgs/network_1_50m/river_label_1/']

    assert len(img_dirs) == len(label_dirs)

    images = []
    labels = []

    print("Starting data preprocessing.")

    for i in range(len(img_dirs)):
        imgs, lbs = get_examples(img_dirs[i], label_dirs[i])
        images.extend(imgs)
        labels.extend(lbs)

    print("Finished preprocessing. Starting training.")

    # initialize model
    model = ConvToLinear(images[0].shape[0])

    # Separate data into train and test
    # proportion of data to retain as training data
    train_ratio = 0.7
    train_idx = math.floor(len(images) * train_ratio)

    train_x = images[:train_idx]
    train_y = labels[:train_idx]

    test_x = images[train_idx:]
    test_y = images[train_idx:]

    # train
    train(model, train_x, train_y)

    # test/return results
    print("Testing...")
    avg_test_loss = test(model, test_x, test_y)
    print(f"FINAL AVERAGE TEST LOSS: {avg_test_loss}")

    # TODO save weights?

    # TODO visualize results?





if __name__ == '__main__':
    main()