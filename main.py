import tensorflow as tf
from model import logDcoral
from solver import Solver

flags = tf.app.flags
# flags.DEFINE_string('mode', 'test', "'train', or 'test'")
flags.DEFINE_string('mode', 'train', "'train', or 'test'")
flags.DEFINE_string('method', 'recon',
                    "the regularizer: 'baseline' (no regularizer), 'd-coral', 'log-d-coral' or 'entropy'")
flags.DEFINE_string('model_save_path', 'model', "base directory for saving the models")
# flags.DEFINE_string('device', '/cpu:0', "/gpu:id number")
flags.DEFINE_string('device', '/gpu:0', "/gpu:id number")
flags.DEFINE_string('alpha', '2.0', "coral regularizer weigtht")
flags.DEFINE_string('beta', '0.15', "reconstruct regularizer weigtht")
flags.DEFINE_string('gamma', '0.1', "generator loss weight")
flags.DEFINE_string('source', 'svhn', 'source doamin')
flags.DEFINE_string('target', 'mnist', 'target doamin')
flags.DEFINE_boolean('reduce', False,'FOR MINIST AND USPS TASK SHOULD REDUCE')
flags.DEFINE_boolean('bn', False, "use batch norm?")
flags.DEFINE_integer('phase',1,"PHASE")
flags.DEFINE_float('T', 1.0, "Temperature")
flags.DEFINE_integer('seed',13,"SEED")
FLAGS = flags.FLAGS

# svhn-->mnist:8,0.0.25,no batch norm,T=4,200000,1800
# usps-->mnist:8,0.25,batch norm,T=2,10000,500
# mnist-->usps:8,0.25,batch norm,T=2,10000,500

# svhn-->mnist 10,0.15,0.1,1,ADAM,0.981
# svn-svhn 8.0,0.01,0.1 ADAM

def main(_):
    with tf.device(FLAGS.device):
        tf.set_random_seed(FLAGS.seed)
        model_save_path = FLAGS.model_save_path + '/' + FLAGS.source+'->'+FLAGS.target+'/'+FLAGS.method + '/alpha_' + FLAGS.alpha+"_beta_"+FLAGS.beta+"_gamma_"+FLAGS.gamma
        log_dir = 'logs/' + FLAGS.source+'->'+FLAGS.target+'/'+FLAGS.method + '/alpha_' + FLAGS.alpha+"_beta_"+FLAGS.beta+"_gamma_"+FLAGS.gamma
        model = logDcoral(mode=FLAGS.mode, method=FLAGS.method, hidden_size=64, learning_rate=0.0001,
                          alpha=float(FLAGS.alpha), beta=float(FLAGS.beta),phase=FLAGS.phase,bn=FLAGS.bn, T=FLAGS.T, gamma=float(FLAGS.gamma))
        solver = Solver(model, batch_size=128, model_save_path=model_save_path, log_dir=log_dir, train_iter=20000, source=FLAGS.source, target=FLAGS.target,reduced=FLAGS.reduce, seed=FLAGS.seed)

        # create directory if it does not exist
        if not tf.gfile.Exists(model_save_path):
            tf.gfile.MakeDirs(model_save_path)

        if FLAGS.mode == 'train':
            solver.train()
        elif FLAGS.mode == 'test':
            solver.test()
        elif FLAGS.mode == 'tsne':
            solver.tsne()
        else:
            print 'Unrecognized mode.'


if __name__ == '__main__':
    tf.app.run()
