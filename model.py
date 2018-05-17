import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

n_l1 = 1000
n_l2 = 1000

class logDcoral(object):
    def __init__(self, mode='training', method='baseline', hidden_size=128, learning_rate=0.0001, batch_size=256,
                 alpha=1.0, beta=0.3, gamma=1.0, T=4.0, phase=1, bn=False):
        self.mode = mode
        self.method = method
        self.learning_rate = learning_rate
        self.hidden_repr_size = hidden_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.T =T
        self.phase = phase
        self.bn=bn
        self.gamma = gamma

    # def E(self, images, is_training=False, reuse=False, bn=False, source=True):
    #
    #     if images.get_shape()[3] == 3:
    #         images = tf.image.rgb_to_grayscale(images)
    #
    #     with tf.variable_scope('encoder', reuse=reuse):
    #         with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
    #             with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
    #                 images = slim.instance_norm(images)
    #                 net = slim.conv2d(images, 64, 5, scope='conv1')
    #                 if bn:
    #                     net = slim.batch_norm(net, is_training=is_training)
    #                 net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
    #                 net = slim.conv2d(net, 128, 5, scope='conv2')
    #                 if bn:
    #                     net = slim.batch_norm(net, is_training=is_training)
    #                 net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
    #                 net = tf.contrib.layers.flatten(net)
    #                 net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc3')
    #                 net = slim.dropout(net, 0.5, is_training=is_training)
    #                 hidden = slim.fully_connected(net, self.hidden_repr_size, activation_fn=tf.nn.relu, scope='fc4')
    #                 z = slim.fully_connected(hidden, 64, activation_fn=tf.identity, scope='latent_codes')
    #                 cat = slim.fully_connected(hidden, 10, activation_fn=tf.identity)
    #                 if not source:
    #                     cat = tf.nn.softmax(cat)
    #                 # dropout here or not?
    #                 # ~ net = slim.dropout(net, 0.5, is_training=is_training)
    #                 return hidden, z, cat, images

    def E(self, images, is_training=False, reuse=False, bn=False, source=True):

        # if images.get_shape()[3] == 3:
        #     images = tf.image.rgb_to_grayscale(images)

        with tf.variable_scope('encoder', reuse=reuse):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
                    images = slim.instance_norm(images)
                    net = slim.conv2d(images, 64, 5, scope='conv1')
                    if bn:
                        net = slim.batch_norm(net, is_training=is_training)
                    net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
                    net = slim.conv2d(net, 128, 5, scope='conv2')
                    if bn:
                        net = slim.batch_norm(net, is_training=is_training)
                    net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
                    net = tf.contrib.layers.flatten(net)
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc3')
                    net = slim.dropout(net, 0.5, is_training=is_training)
                    hidden = slim.fully_connected(net, self.hidden_repr_size, activation_fn=tf.nn.relu, scope='fc4')
                    cat = slim.fully_connected(hidden, 10, activation_fn=tf.identity)
                    if not source:
                        cat = tf.nn.softmax(cat)
                    # dropout here or not?
                    # ~ net = slim.dropout(net, 0.5, is_training=is_training)
                    return hidden, 0, cat, images

    def D(self, repre, y, is_training=False, reuse=False, bn=False):
        with tf.variable_scope('decoder', reuse=reuse):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
                    net = repre#tf.concat([y, repre], 1)
                    net = slim.fully_connected(net, 1024, scope='defc4')
                    net = slim.dropout(net, 0.5, is_training=is_training)
                    net = slim.fully_connected(net, 8 * 8 * 128, scope='defc3')
                    net = tf.reshape(net, [-1, 8, 8, 128])
                    if bn:
                        net = slim.batch_norm(net, is_training=is_training)
                    net = slim.conv2d_transpose(net, 64, (5, 5), 2, activation_fn=tf.nn.relu, scope='deconv2')
                    if bn:
                        net = slim.batch_norm(net, is_training=is_training)
                    net = slim.conv2d_transpose(net, 1, (5, 5), 2, activation_fn=tf.sigmoid, scope='deconv1')
                    return net

    def discriminator_categorical(self, x, reuse=False):
        """
        Discriminator that is used to match the posterior distribution with a given categorical distribution.
        :param x: tensor of shape [batch_size, n_labels]
        :param reuse: True -> Reuse the discriminator variables,
                      False -> Create or search of variables before creating
        :return: tensor of shape [batch_size, 1]
        """
        with tf.variable_scope('Discriminator_Categorial', reuse=reuse):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                dc_den1 = slim.fully_connected(x, n_l1, scope='dc_c_den1')
                dc_den2 = slim.fully_connected(dc_den1, n_l2, scope='dc_c_den2')
                output = slim.fully_connected(dc_den2, 1, scope='dc_c_output', activation_fn=tf.identity)
        return output


    def logits(self, inputs, is_training=False, reuse=False):

        with tf.variable_scope('logits', reuse=reuse):
            with slim.arg_scope([slim.fully_connected], activation_fn=None):
                return slim.fully_connected(inputs, 10, activation_fn=None, scope='fc5')

    def coral_loss(self, h_src, h_trg, gamma=1e-3):
        # regularized covariances (D-Coral is not regularized actually..)
        # First: subtract the mean from the data matrix
        batch_size = tf.to_float(tf.shape(h_src)[0])
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        # Returns the Frobenius norm (there is an extra 1/4 in D-Coral actually)
        # The reduce_mean account for the factor 1/d^2
        return tf.reduce_mean(tf.square(tf.subtract(cov_source, cov_target)))

    def log_coral_loss(self, h_src, h_trg, gamma=1e-3):
        # regularized covariances result in inf or nan
        # First: subtract the mean from the data matrix
        batch_size = tf.to_float(tf.shape(h_src)[0])
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        # eigen decomposition
        eig_source = tf.self_adjoint_eig(cov_source)
        eig_target = tf.self_adjoint_eig(cov_target)
        log_cov_source = tf.matmul(eig_source[1],
                                   tf.matmul(tf.diag(tf.log(eig_source[0])), eig_source[1], transpose_b=True))
        log_cov_target = tf.matmul(eig_target[1],
                                   tf.matmul(tf.diag(tf.log(eig_target[0])), eig_target[1], transpose_b=True))

        # Returns the Frobenius norm
        return tf.reduce_mean(tf.square(tf.subtract(log_cov_source, log_cov_target)))

    # ~ return tf.reduce_mean(tf.reduce_max(eig_target[0]))
    # ~ return tf.to_float(tf.equal(tf.count_nonzero(h_src), tf.count_nonzero(h_src)))


    def build_model(self):

        if self.mode == 'train':
            # self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
            # self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
            self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 1],"source")
            self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1],"target")
            self.val_images = tf.placeholder(tf.float32, [None, 32, 32, 1],"validation")
            self.src_labels = tf.placeholder(tf.int64, [None])
            self.trg_labels = tf.placeholder(tf.int64, [None])
            self.val_labels = tf.placeholder(tf.int64, [None])
            self.categorial_distribution = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='Categorical_distribution')

            self.src_hidden, _, self.src_logits, _ = self.E(self.src_images, is_training=True, bn=self.bn, source=True)
            self.trg_hidden, self.trg_z, self.trg_logits, t_img = self.E(self.trg_images, is_training=True, reuse=True, bn=self.bn, source=False)
            _, _, self.val_logits, _ = self.E(self.val_images, is_training=False, reuse=True, bn=self.bn, source=False)

            # last fc layer to logits
            # self.src_logits = self.logits(self.src_hidden)
            # self.trg_logits = self.logits(self.trg_hidden, reuse=True)
            # self.val_logits = self.logits(self.val_hidden, reuse=True)

            # self.src_recon = self.D(self.src_hidden, tf.one_hot(self.src_labels, 10), softmax=False, is_training=True)
            self.trg_recon = self.D(self.trg_hidden, self.trg_logits, is_training=True, bn=self.bn)

            # discriminator
            self.d_c_real = self.discriminator_categorical(self.categorial_distribution)
            self.d_c_fake = self.discriminator_categorical(self.trg_logits, reuse=True)

            # class predictions
            self.src_pred = tf.argmax(self.src_logits, 1)
            self.src_correct_pred = tf.equal(self.src_pred, self.src_labels)
            self.src_accuracy = tf.reduce_mean(tf.cast(self.src_correct_pred, tf.float32))
            self.trg_pred = tf.argmax(self.trg_logits, 1)
            self.trg_correct_pred = tf.equal(self.trg_pred, self.trg_labels)
            self.trg_accuracy = tf.reduce_mean(tf.cast(self.trg_correct_pred, tf.float32))
            self.val_pred = tf.argmax(self.val_logits, 1)
            self.val_correct_pred = tf.equal(self.val_pred, self.val_labels)
            self.val_accuracy = tf.reduce_mean(tf.cast(self.val_correct_pred, tf.float32))

            # losses: class, domain, total
            self.class_loss = slim.losses.sparse_softmax_cross_entropy(self.src_logits, self.src_labels)

            # discriminator loss
            self.dc_c_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_c_real), logits=self.d_c_real))
            self.dc_c_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.d_c_fake), logits=self.d_c_fake))
            self.dc_c_loss = self.dc_c_loss_fake + self.dc_c_loss_real

            # generator loss
            self.generator_c_loss = self.gamma*tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_c_fake), logits=self.d_c_fake))


            self.trg_softmax = slim.softmax(self.trg_logits/self.T)
            self.trg_entropy = -tf.reduce_mean(tf.reduce_sum(self.trg_softmax * tf.log(self.trg_softmax), axis=1))

            if self.method == 'log-d-coral':
                print('----------------')
                print('| log-d-coral', self.alpha)
                print('----------------')
                self.domain_loss = self.alpha * self.log_coral_loss(self.src_hidden, self.trg_hidden)
                self.loss = self.class_loss + self.domain_loss

            elif self.method == 'd-coral':
                print('----------------')
                print('| d-coral', self.alpha)
                print('----------------')
                self.domain_loss = self.alpha * self.coral_loss(self.src_hidden, self.trg_hidden)
                self.loss = self.class_loss + self.domain_loss

            elif self.method == 'baseline':
                print('----------------')
                print('| baseline')
                print('----------------')
                self.domain_loss = self.alpha * self.coral_loss(self.src_hidden, self.trg_hidden)
                self.loss = self.class_loss

            elif self.method == 'entropy':
                print('----------------')
                print('| entropy', self.alpha)
                print('----------------')
                self.domain_loss = self.alpha * self.trg_entropy
                self.loss = self.class_loss + self.domain_loss
            elif self.method == 'recon':
                if self.phase==1:
                    print('| recon', self.alpha, self.beta, self.gamma, self.phase)
                    self.recon_loss = self.beta * tf.losses.mean_squared_error(self.trg_recon, t_img)
                    self.domain_loss = self.alpha * self.coral_loss(self.src_hidden, self.trg_hidden)
                    self.loss = self.class_loss + self.domain_loss + self.recon_loss+self.generator_c_loss
                elif self.phase==2:
                    print('| recon', self.alpha, self.beta, self.phase)
                    self.recon_loss =  slim.losses.mean_squared_error(self.trg_recon, t_img)
                    self.domain_loss = self.alpha * self.coral_loss(self.src_hidden, self.trg_hidden)
                    self.loss = 1e-3*self.trg_entropy + self.recon_loss


            else:
                print('Unrecognized method')


            self.update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            all_variables = tf.trainable_variables()
            self.en_var = [var for var in all_variables if 'enc' in var.name]
            self.dc_c_var = [var for var in all_variables if 'Discriminator_Categorial' in var.name]
            with tf.control_dependencies(self.update_op):
                # self.train_op = tf.train.MomentumOptimizer(self.learning_rate*10, momentum=0.1,use_nesterov=True).minimize(self.loss)
                # self.discriminator_c_optimizer_op = tf.train.MomentumOptimizer(learning_rate=self.learning_rate*10, momentum=0.1,use_nesterov=True).minimize(self.dc_c_loss, var_list=self.dc_c_var)
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                self.discriminator_c_optimizer_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.dc_c_loss, var_list=self.dc_c_var)
                # self.generator_optimizer_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.generator_c_loss, var_list=self.en_var)

            # summary op
            class_loss_summary = tf.summary.scalar('classification_loss', self.class_loss)
            domain_loss_summary = tf.summary.scalar('domain_loss', self.domain_loss)
            if self.method == "recon":
                recon_loss_summary = tf.summary.scalar('recon loss', self.recon_loss)
            src_accuracy_summary = tf.summary.scalar('src_accuracy', self.src_accuracy)
            trg_accuracy_summary = tf.summary.scalar('trg_accuracy', self.trg_accuracy)
            val_accuracy_summary = tf.summary.scalar('val_accuracy', self.val_accuracy)
            trg_entropy_summary = tf.summary.scalar('trg_entropy', self.trg_entropy)
            if self.method == 'recon':
                self.summary_op = tf.summary.merge([class_loss_summary,
                                                    domain_loss_summary,
                                                    recon_loss_summary,
                                                    src_accuracy_summary,
                                                    trg_accuracy_summary,
                                                    trg_entropy_summary])
                self.val_summary_op = tf.summary.merge([val_accuracy_summary])
            else:
                self.summary_op = tf.summary.merge([class_loss_summary,
                                                    domain_loss_summary,
                                                    src_accuracy_summary,
                                                    trg_entropy_summary])
                self.val_summary_op = tf.summary.merge([val_accuracy_summary])

        elif self.mode == 'test':
            self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'usps_images')
            # self.trg_images = tf.image.resize_images(self.trg_images_32, (28, 28))
            self.trg_labels = tf.placeholder(tf.int64, [None], 'usps_labels')

            _ ,_,self.trg_softmax, _= self.E(self.trg_images, is_training=False, bn=self.bn, source=False)

            # last fc layer to logits
            # self.trg_logits = self.logits(self.trg_hidden)
            # self.trg_softmax = slim.softmax(self.trg_logits)
            self.trg_entropy = -tf.reduce_mean(tf.reduce_sum(self.trg_softmax * tf.log(self.trg_softmax), axis=1))

            self.trg_pred = tf.argmax(self.trg_softmax, 1)
            self.trg_correct_pred = tf.equal(self.trg_pred, self.trg_labels)
            self.trg_accuracy = tf.reduce_mean(tf.cast(self.trg_correct_pred, tf.float32))

        elif self.mode == 'tsne':
            self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
            self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'svhn_images')

            self.trg_hidden,_ = self.E(self.trg_images, is_training=False, bn=self.bn)
            self.src_hidden,_ = self.E(self.src_images, is_training=False, reuse=True, bn=self.bn)

        else:
            print('Unrecognized mode')
