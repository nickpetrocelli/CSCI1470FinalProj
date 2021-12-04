import numpy as np
import tensorflow as tf
from get_data import get_examples
import math
import imageio as iio


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
        self.logit_layer = tf.keras.layers.Dense(image_dim * image_dim, activation='sigmoid')

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

    def loss(self, sig_prob_batch, labels_batch):
        """
        output_batch: BATCH_SIZE x 10,000
        sig_prob_batch: BATCH_SIZE x 100 x 100
        """
        labels = tf.convert_to_tensor(labels_batch, dtype=tf.float64)
        labels_flat = tf.reshape(labels, (labels.shape[0], self.image_dim * self.image_dim))

        #idea: sum of crossentropy loss
        # partially inspired by hw5, this is somewhat of a "reconstruction loss" of the output river network
        # though this is a classification problem, not a generation one.
        loss_calc = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, 
            reduction=tf.keras.losses.Reduction.SUM,
        )
        loss = loss_calc(labels_flat, sig_prob_batch)
        return loss


def train(model, train_images, train_labels, epoch, batch_size = 100):
    # epoch is just for printing    

    batch_index = 0
    batch_num = 0
    while batch_index < len(train_images):
        input_batch = train_images[batch_index:batch_index+batch_size]
        label_batch = train_labels[batch_index:batch_index+batch_size]

        with tf.GradientTape() as tape:
            sig_probs = model.call(input_batch)
            loss = model.loss(sig_probs, label_batch)

            print(f"TRAIN|{epoch}|{batch_num}|{loss}|")

        # from lab
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        batch_index = batch_index + batch_size
        batch_num = batch_num + 1

    return 

def test(model, test_images, test_labels, epoch, batch_size = 100):
    # very similar to homework procedure
    accuracy_accumulator = tf.keras.metrics.Accuracy()
    # TODO this should work but it doesn't for some reason
    #accuracy_accumulator.reset_state()
    
    # TODO need a better accuracy metric
    # currently just return average test loss

    losses = []
    batch_index = 0
    batch_num = 0
    while batch_index < len(test_images):
        input_batch = test_images[batch_index:batch_index+batch_size]
        label_batch = test_labels[batch_index:batch_index+batch_size]

        sig_probs = model.call(input_batch)
        loss = model.loss(sig_probs, label_batch)

        # have to do some weird transformations to leverage keras' accuracy func
        # see https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Accuracy
        # essentially taking highest probability class from outputs then comparing to labels
        transformed_probs = tf.reshape(sig_probs, [-1])
        # TODO there is probably a much better way to do this with tf.map_fn but I couldn't figure one out
        transformed_probs = tf.convert_to_tensor([1 if x>0.5 else 0 for x in transformed_probs])
        transformed_probs = tf.reshape(transformed_probs, [-1, 1])
        transformed_labels = tf.reshape(label_batch, [-1, 1])
        accuracy_accumulator.update_state(y_true=transformed_labels, y_pred=transformed_probs)


        print(f"TEST|{epoch}|{batch_num}|{loss}|{accuracy_accumulator.result()}")

        losses.append(loss)
        batch_index = batch_index + batch_size
        batch_num = batch_num + 1

    # return accuracy
    return accuracy_accumulator.result()


def output_to_imgarray(model, input_image):
    """
    Visualizes the output for a given image
    :param input_image: 1x100x100x1 image input
    :return 100x100x4 numpy array representing .png (can be output using imageio)
    """
    # get sigmoid probabilities
    sig_probs = model.call(input_image)
    
    output_flat = []
    for i in sig_probs[0]:
        if i > tf.convert_to_tensor(0.5):
            output_flat.append([0, 0, 0, 255])
        else:
            output_flat.append([0, 0, 0, 0])

    output = np.array(output_flat, dtype=np.uint8)
    output = np.reshape(output, (model.image_dim, model.image_dim, 4))

    return output


def visualize_imgarray(img_array, filename='output.png', directory='../outputs'):
    """
    Writes the given image to the given filename in the given directory
    :param img_array: 100x100x4 numpy array representing .png (output of output_to_imgarray)
    """
    iio.imwrite(f'{directory}/{filename}', img_array)


def main():
    NUM_EPOCHS = 1
    img_dirs = ['../data/imgs/network_1_50m/stream_network_1_buff_50m/']
    label_dirs = ['../data/imgs/network_1_50m/river_label_1/']

    # TODO DEBUG - for consistent results when testing/debugging, should remove for primetime?
    tf.random.set_seed(12345)

    assert len(img_dirs) == len(label_dirs)

    images = []
    labels = []

    #print("Starting data preprocessing.")

    for i in range(len(img_dirs)):
        imgs, lbs = get_examples(img_dirs[i], label_dirs[i])
        images.extend(imgs)
        labels.extend(lbs)


    # shuffle the input as I have in the homework
    # doing this now to improve performance by preventing issues due to
    # training set and testing set coming from disjoint regions of the original image
    shuffled_indices = tf.random.shuffle([i for i in range(len(images))])
    shuffled_inputs = tf.gather(images, shuffled_indices)
    shuffled_labels = tf.gather(labels, shuffled_indices)

    #print("Finished preprocessing. Starting training.")

    # initialize model
    model = ConvToLinear(images[0].shape[0])

    # Separate data into train and test
    # proportion of data to retain as training data
    train_ratio = 0.7
    train_idx = math.floor(len(images) * train_ratio)

    train_x = shuffled_inputs[:train_idx]
    train_y = shuffled_labels[:train_idx]

    test_x = shuffled_inputs[train_idx:]
    test_y = shuffled_labels[train_idx:]

    # TA notes:
    # train only on one image for many epochs, then test on the same one image
    # should tell if weights are getting updated

    # visualize pair of inputs?



    # #TODO DEBUG
    # train_x = [images[5]]
    # train_y = [labels[5]]

    # test_x = train_x
    # test_y = train_y
    # # train
    # for i in range(250):
    #     print(f"epoch {i}")
    #     train(model, train_x, train_y, batch_size = 1)

    print("run_type|epoch_num|batch_num|loss|avg_accuracy")
    for i in range(NUM_EPOCHS):
        #print(f"EPOCH {i}")
        train(model, train_x, train_y, 0)

    # test/return results
    #print("Testing...")
    test_acc = test(model, test_x, test_y, 0)
    #print(f"FINAL TEST ACCURACY: {test_acc}")

    #print("DEBUG: testing on train inputs")
    debug_test_acc = test(model, train_x, train_y, 0)
    #print(f"DEBUG TEST ACCURACY: {debug_test_acc}")

    # TODO save weights?

    # TODO visualize results?
    # should be IMG-1001
    visualize_imgarray(output_to_imgarray(model, [images[5]]), filename='image-1001-output.png')



if __name__ == '__main__':
    main()