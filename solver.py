import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import time

import matplotlib.pyplot as plt
import matplotlib as mpl

import utils
from sklearn.manifold import TSNE
from scipy import misc
import gzip
from skimage.viewer import ImageViewer
from sklearn.utils import shuffle

svhn_dir = 'data/svhn'
mnist_dir = 'data/mnist'
usps_dir = 'data/usps'
mnist_m_dir = 'data/mnist-m'
svn_dir = 'data/SynthDigits'


# ~ from utils import resize_images

class Solver(object):
    def __init__(self, model, batch_size=128, train_iter=100000,
                 source='svhn', target='usps', log_dir='logs',
                 model_save_path='model', trained_model='model/model', reduced=True, seed=0):

        self.model = model
        self.batch_size = batch_size
        self.train_iter = train_iter
        self.svhn_dir = svhn_dir
        self.svn_dir = svn_dir
        self.usps_dir = usps_dir
        self.mnist_dir = mnist_dir
        self.mnist_m_dir = mnist_m_dir
        self.source = source
        self.target = target
        self.log_dir = log_dir
        self.model_save_path = model_save_path
        self.trained_model = model_save_path  # + '/model'
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.reduced = reduced
        self.seed = seed

    def load_mnist(self, image_dir, split='train', seed=0, reduced=False):
        print ('Loading MNIST dataset.')

        image_file = 'mnist.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnist = pickle.load(f)
        if split == 'train':
            images, labels = mnist['training_images'], mnist['training_labels']
        else:
            images, labels = mnist['test_images'], mnist['test_labels']
        if reduced:
            images, labels = shuffle(images, labels, random_state=seed)
            images, labels = images[:2000], labels[:2000]
        # ~ images= resize_images(images)
        # images = images * 2.0 - 1.0
        return np.expand_dims(images, 3), labels

        # return images, np.squeeze(labels).astype(int)

    def load_svhn(self, image_dir, split='train'):
        print ('Loading SVHN dataset.')

        image_file = 'svhn.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            svhn = pickle.load(f)
        if split == 'train':
            images, labels = svhn['training_images'], svhn['training_labels']
        else:
            images, labels = svhn['test_images'], svhn['test_labels']
        # ~ images= resize_images(images)
        # images = images*2.0-1.0
        return np.expand_dims(images, 3), labels

    def load_svn(self, image_dir, split='train'):
        print ('Loading SVN dataset.')

        image_file = 'svn.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            svn = pickle.load(f)
        if split == 'train':
            images, labels = svn['training_images'], svn['training_labels']
        else:
            images, labels = svn['test_images'], svn['test_labels']
        # ~ images= resize_images(images)
        images = images / 255.0
        return np.expand_dims(images, 3), labels

    def load_usps(self, image_dir, split='train', seed=0, reduced=False):
        print ('Loading USPS dataset.')
        image_file = 'usps_32_train.pkl' if split == 'train' else 'usps_32_test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            usps = pickle.load(f)
        f.close()
        if reduced:
            images, labels = shuffle(usps['X'], usps['y'], random_state=seed)
            images, labels = images[:1800], labels[:1800]
        else:
            images, labels = usps['X'], usps['y']

        return images, labels

    def load_mnist_m(self, image_dir, split='train'):
        print ('Loading MNIST-M dataset.')
        image_file = 'mnistm_data.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnistm = pickle.load(f)
        f.close()
        if split == 'train':
            images, labels = mnistm['train_images'], mnistm['train_labels']
        else:
            images, labels = mnistm['test_images'], mnistm['test_labels']
        return images, labels

    def train(self):

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        print('Training.')

        # Data loading
        if self.source == 'mnist':
            src_images, src_labels = self.load_mnist(self.mnist_dir, split='train', reduced=self.reduced,
                                                     seed=self.seed)
            src_test_images, src_test_labels = self.load_mnist(self.mnist_dir, split='test')
        elif self.source == 'usps':
            src_images, src_labels = self.load_usps(self.usps_dir, split='train', reduced=self.reduced, seed=self.seed)
            src_test_images, src_test_labels = self.load_usps(self.usps_dir, split='test')
        elif self.source == 'svhn':
            src_images, src_labels = self.load_svhn(self.svhn_dir, split='train')
            src_test_images, src_test_labels = self.load_svhn(self.svhn_dir, split='test')
        elif self.source == 'svn':
            src_images, src_labels = self.load_svn(self.svn_dir, split='train')
            src_test_images, src_test_labels = self.load_svn(self.svn_dir, split='test')
        elif self.source == 'mnist_m':
            src_images, src_labels = self.load_mnist_m(self.mnist_m_dir, split='train')
            src_test_images, src_test_labels = self.load_mnist_m(self.mnist_m_dir, split='test')

        if self.target == 'mnist':
            trg_images, trg_labels = self.load_mnist(self.mnist_dir, split='train', reduced=self.reduced,
                                                     seed=self.seed)
            trg_test_images, trg_test_labels = self.load_mnist(self.mnist_dir, split='test')
        elif self.target == 'usps':
            trg_images, trg_labels = self.load_usps(self.usps_dir, split='train', reduced=self.reduced, seed=self.seed)
            trg_test_images, trg_test_labels = self.load_usps(self.usps_dir, split='test')
        elif self.target == 'svhn':
            trg_images, trg_labels = self.load_svhn(self.svhn_dir, split='train')
            trg_test_images, trg_test_labels = self.load_svhn(self.svhn_dir, split='test')
        elif self.target == 'svn':
            trg_images, trg_labels = self.load_svn(self.svn_dir, split='train')
            trg_test_images, trg_test_labels = self.load_svn(self.svn_dir, split='test')
        elif self.target == 'mnist_m':
            trg_images, trg_labels = self.load_mnist_m(self.mnist_m_dir, split='train')
            trg_test_images, trg_test_labels = self.load_mnist_m(self.mnist_m_dir, split='test')

        # build a graph
        model = self.model
        model.build_model()

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(max_to_keep=5)

            start_step = 0
            ckpt = tf.train.get_checkpoint_state(self.trained_model)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                start_step = int(ckpt.model_checkpoint_path.split("-")[2])
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

            print ('Start training.')
            trg_count = 0
            t = 0
            start_time = time.time()
            for step in range(start_step, self.train_iter):

                trg_count += 1
                t = step

                i = step % int(src_images.shape[0] / self.batch_size)
                j = step % int(trg_images.shape[0] / self.batch_size)

                # generate category distribution
                real_cat_dist = np.random.randint(low=0, high=10, size=self.batch_size)
                real_cat_dist = np.eye(10)[real_cat_dist]

                feed_dict = {model.src_images: src_images[i * self.batch_size:(i + 1) * self.batch_size],
                             model.src_labels: src_labels[i * self.batch_size:(i + 1) * self.batch_size],
                             model.trg_images: trg_images[j * self.batch_size:(j + 1) * self.batch_size],
                             model.trg_labels: trg_labels[j * self.batch_size:(j + 1) * self.batch_size],
                             model.categorial_distribution: real_cat_dist
                             }

                sess.run(model.train_op, feed_dict)
                # if t%1==0: #train g twice
                sess.run(model.discriminator_c_optimizer_op, feed_dict)
                # sess.run(model.generator_optimizer_op, feed_dict)
                # sess.run(model.train_op, feed_dict)
                if t % 100 == 0:
                    summary, l_c, l_d, l_r, l_g, l_dis,src_acc = sess.run(
                        [model.summary_op, model.class_loss, model.domain_loss, model.recon_loss, model.generator_c_loss, model.dc_c_loss, model.src_accuracy],
                        feed_dict)
                    summary_writer.add_summary(summary, t)
                    print ('Step: [%d/%d]  c_loss: [%.6f]  d_loss: [%.6f] recon_loss: [%.6f] gen_loss: [%.6f] dis_loss: [%.6f] train acc: [%.4f]' \
                           % (t, self.train_iter, l_c, l_d, l_r, l_g, l_dis, src_acc))
                if t % 500 == 0:
                    val_feed_dict = {model.val_images: trg_test_images[:1000],
                                     model.val_labels: trg_test_labels[:1000]}
                    summary, val_acc = sess.run(
                        [model.val_summary_op, model.val_accuracy], val_feed_dict)
                    summary_writer.add_summary(summary, t)
                    saver.save(sess, os.path.join(self.model_save_path, 'model'), global_step=t)
                    print ('Step: [%d/%d] val acc: [%.4f]' \
                           % (t, self.train_iter, val_acc))
                    # l_c,l_r,l_l = sess.run([model.trg_entropy, model.recon_loss, model.loss], feed_dict)
                    # print ('Step: [%d/%d]  c_loss: [%.6f]  recon_loss: [%.6f] all loss: [%.6f]' \
                    #        % (t, self.train_iter, l_c,l_r,l_l))

                    # ~ if t%10000==0:
                    # ~ print 'Saved.'
            with open('time_' + str(model.alpha) + '_' + model.method + '.txt', "a") as resfile:
                resfile.write(str((time.time() - start_time) / float(self.train_iter)) + '\n')
            saver.save(sess, os.path.join(self.model_save_path, 'model'), global_step=t)

    def test(self):
        if self.target == "mnist":
            trg_images, trg_labels = self.load_mnist(self.mnist_dir, split='test')
        elif self.target == "svhn":
            trg_images, trg_labels = self.load_svhn(self.svhn_dir, split='test')
        elif self.target == "usps":
            trg_images, trg_labels = self.load_usps(self.usps_dir, split='test')

        # build a graph
        model = self.model
        model.build_model()

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()

            ckpt = tf.train.get_checkpoint_state(self.trained_model)
            print ('Loading  model.', ckpt.model_checkpoint_path)
            variables_to_restore = slim.get_model_variables()
            restorer = tf.train.Saver(variables_to_restore)

            restorer.restore(sess, ckpt.model_checkpoint_path)

            trg_acc, trg_entr = sess.run(fetches=[model.trg_accuracy, model.trg_entropy],
                                         feed_dict={model.trg_images: trg_images[trg_images.shape[0] // 2:],
                                                    model.trg_labels: trg_labels[trg_labels.shape[0] // 2:]})

            print ('test acc [%.3f]' % (trg_acc))
            print ('entropy [%.3f]' % (trg_entr))
            with open('test_' + str(model.alpha) + '_' + model.method + '.txt', "a") as resfile:
                resfile.write(str(trg_acc) + '\t' + str(trg_entr) + '\n')

                # ~ print confusion_matrix(trg_labels, trg_pred)

    def tsne(self, n_samples=2000):
        if self.source == 'mnist':
            src_images, src_labels = self.load_mnist(self.mnist_dir, split='test')
        elif self.source == 'usps':
            src_images, src_labels = self.load_usps(self.usps_dir, split='test')
        elif self.source == 'svhn':
            src_images, src_labels = self.load_svhn(self.svhn_dir, split='test')

        if self.target == "mnist":
            trg_images, trg_labels = self.load_mnist(self.mnist_dir, split='test')
        elif self.target == "svhn":
            trg_images, trg_labels = self.load_svhn(self.svhn_dir, split='test')
        elif self.target == "usps":
            trg_images, trg_labels = self.load_usps(self.usps_dir, split='test')

        model = self.model
        model.build_model()

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(self.trained_model)
            print ('Loading  model.', ckpt.model_checkpoint_path)
            variables_to_restore = slim.get_model_variables()
            restorer = tf.train.Saver(variables_to_restore)

            restorer.restore(sess, ckpt.model_checkpoint_path)
            from sklearn.utils import shuffle
            src_images, src_labels = shuffle(src_images, src_labels,random_state=0)
            trg_images, trg_labels = shuffle(trg_images, trg_labels,random_state=0)


            target_images = trg_images[:n_samples]
            target_labels = trg_labels[:n_samples]
            source_images = src_images[:n_samples]
            source_labels = src_labels[:n_samples]
            print(source_labels.shape)

            assert len(target_labels) == len(source_labels)

            src_labels = utils.one_hot(source_labels.astype(int), 10)
            trg_labels = utils.one_hot(target_labels.astype(int), 10)

            n_slices = int(n_samples / self.batch_size)

            fx_src = np.empty((0, model.hidden_repr_size))
            fx_trg = np.empty((0, model.hidden_repr_size))

            for src_im, trg_im in zip(np.array_split(source_images, n_slices),
                                      np.array_split(target_images, n_slices),
                                      ):
                feed_dict = {model.src_images: src_im, model.trg_images: trg_im}

                fx_src_, fx_trg_ = sess.run([model.src_hidden, model.trg_hidden], feed_dict)

                fx_src = np.vstack((fx_src, np.squeeze(fx_src_)))
                fx_trg = np.vstack((fx_trg, np.squeeze(fx_trg_)))

            src_labels = np.argmax(src_labels, 1)
            trg_labels = np.argmax(trg_labels, 1)

            assert len(src_labels) == len(fx_src)
            assert len(trg_labels) == len(fx_trg)

            print('Computing T-SNE.')

            model = TSNE(n_components=2, random_state=0)

            TSNE_hA = model.fit_transform(np.vstack((fx_src, fx_trg)))
            plt.figure(2)
            plt.scatter(TSNE_hA[:, 0], TSNE_hA[:, 1], c=np.hstack((src_labels, trg_labels,)), s=3, cmap=mpl.cm.jet)
            plt.figure(3)
            plt.scatter(TSNE_hA[:, 0], TSNE_hA[:, 1], c=np.hstack((np.ones((n_samples,)), 2 * np.ones((n_samples,)))),
                        s=3, cmap=mpl.cm.jet)

            np.save("cdmda.npy", TSNE_hA)
            plt.show()


if __name__ == '__main__':
    image_dir = '/tmp/data/usps'
    image_file = "usps_28x28.pkl"
    image_dir = os.path.join(image_dir, image_file)
    f = gzip.open(image_dir, "rb")
    usps = pickle.load(f)
    f.close()

    viewer = ImageViewer(np.transpose(usps[0][0][0], (1, 2, 0))[:, :, 0])
    viewer.show()
    print(usps)
    print('empty')
