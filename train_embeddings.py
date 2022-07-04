import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import numpy as np
import os
from utils import *
import random
import time

config = tf.compat.v1.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
a=tf.test.is_gpu_available(
    cuda_only=True, min_cuda_compute_capability=None
)

chunks_folder = 'wikitables/mcon_data'

input_word2int=np.load(chunks_folder+'/input_word2int_with_context.npy',allow_pickle=True)
input_word2int=input_word2int[()]

input_vocab_size = len(input_word2int)

output_word2int=np.load(chunks_folder+'/output_word2int_with_context.npy',allow_pickle=True)
output_word2int=output_word2int[()]

output_vocab_size = len(output_word2int)

input_folder = chunks_folder + '/X'
label_folder = chunks_folder + '/Y'

nb_chunks = len(os.listdir(input_folder))

model_dir = chunks_folder + '/skip_gram_model_with_context'
checkpoint_name = 'model.ckpt'
checkpoint_path = os.path.join(model_dir, checkpoint_name)

EMBEDDING_DIM = 100
num_sampled = 10000
n_iters = 3
batch_size = 100
resume_training=0
chechpoint_period = 200000
print_loss_period=4000

if resume_training:
    nb_steps_per_epoch=steps_per_epoch(chunks_folder,batch_size)


with tf.device("gpu"):
    # making placeholders for x_train and y_train
    x = tf.compat.v1.placeholder(tf.int32, shape=[None])
    y_label = tf.compat.v1.placeholder(tf.int32, shape=[None])
    x_onehot = tf.one_hot(x, input_vocab_size)
    y_onehot = tf.one_hot(y_label, output_vocab_size)

    W1 = tf.Variable(tf.compat.v1.random_normal([input_vocab_size, EMBEDDING_DIM]))
    b1 = tf.Variable(tf.compat.v1.random_normal([EMBEDDING_DIM]))  # bias
    #hidden_representation = tf.add(tf.matmul(x_onehot, W1), b1)
    hidden_representation = tf.matmul(x_onehot, W1)

    # W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
    W2 = tf.Variable(tf.compat.v1.random_normal([output_vocab_size, EMBEDDING_DIM]))
    b2 = tf.Variable(tf.compat.v1.random_normal([output_vocab_size]))
    # prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))


    labels = tf.reshape(y_label, [-1, 1])
    global_step = tf.Variable(0, name='global_step', trainable=False)


    cross_entropy_loss = tf.reduce_mean(tf.nn.nce_loss(
        weights=W2,
        biases=b2,
        labels=labels,
        inputs=hidden_representation,
        num_sampled=num_sampled,
        # num_true=num_true,
        num_classes=output_vocab_size))


    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_loss, global_step=global_step)

    saver = tf.compat.v1.train.Saver(max_to_keep=2)



# bind the graph to session, and run the session

with tf.compat.v1.Session(config=config) as sess:
    try:
        # Restore variables from disk.
        saver.restore(sess, tf.compat.v1.train.latest_checkpoint(model_dir))
    except:
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        saver.save(sess, checkpoint_path)


    # train for n_iter iterations
    glob_step = sess.run(global_step)
    iter = 0

    timestamp1 = time.time()

    avg_time=[]

    for step in range(n_iters):
        if not resume_training:
            for chunk in range(nb_chunks):
                train_x_file = 'x_train_' + str(chunk + 1) + '.npy'
                train_y_file = 'y_train_' + str(chunk + 1) + '.npy'
                train_x_file = os.path.join(input_folder, train_x_file)
                train_y_file = os.path.join(label_folder, train_y_file)
                x_train = np.load(train_x_file)
                y_train = np.load(train_y_file)
                nb_batches = len(x_train) // batch_size
                rest_of_samples = len(x_train) % batch_size

                for cb in range(1, nb_batches + 1):
                    iter += 1
                    glob_step += 1
                    x_train_batch = x_train[batch_size * (cb - 1):batch_size * cb]
                    y_train_batch = y_train[batch_size * (cb - 1):batch_size * cb]

                    if len(x_train_batch)>0:
                        sess.run(train_step, feed_dict={x: x_train_batch, y_label: y_train_batch})

                        if iter % chechpoint_period == 0:
                            saver.save(sess, checkpoint_path, global_step=global_step)

                        if iter %print_loss_period==0:
                            timestamp2 = time.time()
                            print('loss is : ',
                                  sess.run(cross_entropy_loss, feed_dict={x: x_train_batch, y_label: y_train_batch}),
                                  glob_step)

                            processing_time=timestamp2 - timestamp1

                            print("This took %.2f seconds" % (processing_time))
                            avg_time.append(processing_time)
                            print('average of time is '+str(np.mean(avg_time)))
                            timestamp1 = time.time()

                x_train_batch = x_train[batch_size * nb_batches:]
                y_train_batch = y_train[batch_size * nb_batches:]

                if len(x_train_batch) > 0:
                    iter += 1
                    glob_step += 1

                    sess.run(train_step, feed_dict={x: x_train_batch, y_label: y_train_batch})

                    if iter % chechpoint_period == 0:
                        saver.save(sess, checkpoint_path, global_step=global_step)
                    if iter % print_loss_period == 0:
                        print(
                            'loss is : ',
                            sess.run(cross_entropy_loss, feed_dict={x: x_train_batch, y_label: y_train_batch}), glob_step)

            # Save the final model
            saver.save(sess, checkpoint_path, global_step=global_step)
            #print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train_batch, y_label: y_train_batch}),
            #      glob_step)


        else:

            current_chunk, current_sample=find_chunk_and_sample(chunks_folder,batch_size,nb_steps_per_epoch)

            print(current_chunk)
            print(current_sample)

            train_x_file = 'x_train_' + str(current_chunk) + '.npy'
            train_y_file = 'y_train_' + str(current_chunk) + '.npy'
            train_x_file = os.path.join(input_folder, train_x_file)
            train_y_file = os.path.join(label_folder, train_y_file)

            x_train = np.load(train_x_file)
            y_train = np.load(train_y_file)

            x_train=x_train[current_sample::]
            y_train = y_train[current_sample::]

            nb_batches = len(x_train) // batch_size
            rest_of_samples = len(x_train) % batch_size

            for cb in range(1, nb_batches + 1):
                iter += 1
                glob_step += 1
                x_train_batch = x_train[batch_size * (cb - 1):batch_size * cb]
                y_train_batch = y_train[batch_size * (cb - 1):batch_size * cb]
                sess.run(train_step, feed_dict={x: x_train_batch, y_label: y_train_batch})

                if iter % chechpoint_period == 0:
                    saver.save(sess, checkpoint_path, global_step=global_step)

                if iter % print_loss_period == 0:
                    print('loss is : ',
                          sess.run(cross_entropy_loss, feed_dict={x: x_train_batch, y_label: y_train_batch}),
                          glob_step)

            x_train_batch = x_train[batch_size * nb_batches:]
            y_train_batch = y_train[batch_size * nb_batches:]

            if len(x_train_batch) > 0:
                iter += 1
                glob_step += 1

                sess.run(train_step, feed_dict={x: x_train_batch, y_label: y_train_batch})

                if iter % chechpoint_period == 0:
                    saver.save(sess, checkpoint_path, global_step=global_step)
                if iter % print_loss_period == 0:
                    print(
                        'loss is : ',
                        sess.run(cross_entropy_loss, feed_dict={x: x_train_batch, y_label: y_train_batch}), glob_step)



            for chunk in range(current_chunk+1,nb_chunks+1):
                train_x_file = 'x_train_' + str(chunk) + '.npy'
                train_y_file = 'y_train_' + str(chunk) + '.npy'

                train_x_file = os.path.join(input_folder, train_x_file)
                train_y_file = os.path.join(label_folder, train_y_file)

                x_train = np.load(train_x_file)
                y_train = np.load(train_y_file)

                perm = np.random.permutation(len(x_train))
                x_train_perm = x_train[perm]
                y_train_perm = y_train[perm]

                nb_batches = len(x_train) // batch_size
                rest_of_samples = len(x_train) % batch_size

                for cb in range(1, nb_batches + 1):
                    iter += 1
                    glob_step += 1
                    x_train_batch = x_train_perm[batch_size * (cb - 1):batch_size * cb]
                    y_train_batch = y_train_perm[batch_size * (cb - 1):batch_size * cb]
                    sess.run(train_step, feed_dict={x: x_train_batch, y_label: y_train_batch})

                    if iter % chechpoint_period == 0:
                        saver.save(sess, checkpoint_path, global_step=global_step)

                    if iter %print_loss_period==0:
                        print('loss is : ',
                              sess.run(cross_entropy_loss, feed_dict={x: x_train_batch, y_label: y_train_batch}),
                              glob_step)

                x_train_batch = x_train[batch_size * nb_batches:]
                y_train_batch = y_train[batch_size * nb_batches:]

                if len(x_train_batch) > 0:
                    iter += 1
                    glob_step += 1

                    sess.run(train_step, feed_dict={x: x_train_batch, y_label: y_train_batch})

                    if iter % chechpoint_period == 0:
                        saver.save(sess, checkpoint_path, global_step=global_step)
                    if iter % print_loss_period == 0:
                        print(
                            'loss is : ',
                            sess.run(cross_entropy_loss, feed_dict={x: x_train_batch, y_label: y_train_batch}), glob_step)



            # Save the final model
            saver.save(sess, checkpoint_path, global_step=global_step)
            #print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train_batch, y_label: y_train_batch}),
            #      glob_step)

print('done')
