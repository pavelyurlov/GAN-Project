import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras import layers as L
from functools import partial


def get_tf_dataset(dataset, batch_size):
    N_EXAMPLES = len(dataset)
    shuffler = tf.contrib.data.shuffle_and_repeat(N_EXAMPLES)
    dataset_tf = tf.data.Dataset.from_tensor_slices(dataset)
    suffled_ds = shuffler(dataset_tf)
    
    dataset_final = suffled_ds.batch(batch_size).prefetch(1)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset_final)
    return iterator.get_next()

def sample_noise_batch(bsize, X_DIMS):
    return np.random.normal(size=(bsize, X_DIMS)).astype('float32')


class Model:
    def fit(self, X, Y, n_hidden_layers=6, output_dim=128, n_epochs=4e4):
        sess = tf.InteractiveSession()
        
        GAN_TYPE = "Jensen-Shannon"
        X_DIMS, Y_DIMS = 3, 5

        generator_activation = keras.activations.elu

        with tf.name_scope("Generator"):
            generator = Sequential(name="Generator")
            generator.add(L.InputLayer([2 * X_DIMS], name='noise'))
            for i in range(n_hidden_layers):
                generator.add(L.Dense(output_dim, activation=generator_activation))
            generator.add(L.Dense(Y_DIMS, activation=None))

        discriminator_activation = partial(keras.activations.relu, alpha=0.3)
        with tf.name_scope("Discriminator"):
            discriminator = Sequential(name="Discriminator")
            discriminator.add(L.InputLayer([Y_DIMS]))
            for i in range(n_hidden_layers):
                discriminator.add(L.Dense(output_dim, activation=discriminator_activation))         
            if GAN_TYPE == "Jensen-Shannon":
                discriminator.add(L.Dense(2, activation=tf.nn.log_softmax))

        train_batch_size = int(1e3)
        real_data = get_tf_dataset(Y, train_batch_size)
        real_data = tf.dtypes.cast(real_data, tf.float32)
        discriminator_real = discriminator(real_data)


        noise_batch_size = tf.placeholder(tf.int32, shape=[], name="noise_batch_size")
        noise = tf.random_normal([noise_batch_size, X_DIMS], dtype=tf.float32, name="noise")
        X_fetched = get_tf_dataset(X, train_batch_size)
        X_fetched = tf.dtypes.cast(X_fetched, tf.float32)
        generated_data = generator(tf.concat([X_fetched, noise], axis=1))
        
        logp_real = discriminator(real_data)
        logp_gen = discriminator(generated_data)
        discriminator_loss = -tf.reduce_mean(logp_real[:,1] + logp_gen[:,0])
        
        disc_learning_rate = 1e-3
        disc_optimizer = tf.train.GradientDescentOptimizer(disc_learning_rate).minimize(
                discriminator_loss, var_list=discriminator.trainable_weights)
        
        generator_loss = -tf.reduce_mean(logp_gen[:,1])
        
        tf_iter = tf.Variable(initial_value=0, dtype=tf.int32)
        learning_rate = tf.train.exponential_decay(4e-4, tf_iter, 200, 0.98)
        gen_optimizer = tf.group(
            tf.train.AdamOptimizer(learning_rate).minimize(generator_loss, var_list=generator.trainable_weights),
            tf.assign_add(tf_iter, 1))
        
        learning_summary = tf.summary.merge([
            tf.summary.scalar("discriminator_loss", discriminator_loss),
            tf.summary.scalar("generator_loss", generator_loss),
        ])

        sess.run(tf.global_variables_initializer())

        TOTAL_ITERATIONS = int(n_epochs)
        DISCRIMINATOR_ITERATIONS = 5
        for epoch in range(TOTAL_ITERATIONS):
            for i in range(DISCRIMINATOR_ITERATIONS):
                sess.run(disc_optimizer, {noise_batch_size: train_batch_size})
            summary, _, _ = sess.run([learning_summary, gen_optimizer, tf_iter], {noise_batch_size: train_batch_size})
    
        self.generator = generator
        self.y_cols = Y.columns
        self.X_DIMS = X_DIMS
        self.sess = sess

    def predict(self, X): 
        noise = sample_noise_batch(bsize=len(X), X_DIMS=self.X_DIMS)
        Y_pred = generator.predict(np.concatenate([X, noise], axis=1))
        Y_pred = pd.DataFrame(Y_pred)
        Y_pred.columns = self.y_cols
        return Y_pred
