import tensorflow as tf
import os, sys
sys.path.append(os.getcwd())
from FaceAging import FaceAging
import argparse

'''
--Train Phase
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset UTKFace --savedir ./save/xav_e50_lr0.0002 --use_trained_model False --epoch 50
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset UTKFace --savedir ./save/xav_e200_lr0.00025 --use_trained_model False --epoch 200


CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset UTKFace --savedir ./save/tnorm_e200_lr0.0002 --use_trained_model False --epoch 200
CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset UTKFace --savedir ./save/tnorm_e50_lr0.00025 --use_trained_model False --epoch 500

CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset UTKFace --savedir ./save/xav_e80_lr0.0002 --use_trained_model True --epoch 80 --lr 0.0002



--Test Phase Test set 없애버렷
CUDA_VISIBLE_DEVICES=0 python3 main.py --is_train False --testdir ./data/man2baby  --savedir save/ --dir_name xav_e50_lr0.0002
CUDA_VISIBLE_DEVICES=0 python3 main.py --is_train False --testdir ./data/man2baby  --savedir save/ --dir_name xav_e50_lr0.0005


'''
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='CAAE')
parser.add_argument('--is_train', type=str2bool, default=True)
parser.add_argument('--epoch', type=int, default=200, help='number of epochs')
parser.add_argument('--dataset', type=str, default='UTKFace', help='training dataset name that stored in ./data')
parser.add_argument('--savedir', type=str, default='./save', help='dir of saving checkpoints and intermediate training results')
parser.add_argument('--testdir', type=str, default='None', help='dir of testing images')
parser.add_argument('--use_trained_model', type=str2bool, default=True, help='whether train from an existing model or from scratch')
parser.add_argument('--use_init_model', type=str2bool, default=True, help='whether train from the init model if cannot find an existing model')
parser.add_argument('--dir_name', type=str, default='xavier_e50_lr', help='To identify directory, write current time or the other information. ')
parser.add_argument('--lr', type=float, default='0.0002', help='Type Learning rate ')
FLAGS = parser.parse_args()


def main(_):

    # print settings
    import pprint
    pprint.pprint(FLAGS)
    if not os.path.exists(FLAGS.savedir):
        os.mkdir(FLAGS.savedir)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        model = FaceAging(dir_name = FLAGS.dir_name,
                          session = session, 
                          is_training=FLAGS.is_train,  
                          save_dir=FLAGS.savedir,  
                          dataset_name=FLAGS.dataset,
                          learning_rate = FLAGS.lr
                           )
        if FLAGS.is_train:
            print('\n\tTraining Mode')
            if not FLAGS.use_trained_model:
                print('\n\tPre-train the network')
                model.train(
                    num_epochs=10,  # number of epochs
                    use_trained_model=FLAGS.use_trained_model,
                    use_init_model=FLAGS.use_init_model,
                    weigts=(0, 0, 0)
                )
                print('\n\tPre-train is done! The training will start.')
            model.train(
                num_epochs=FLAGS.epoch,  # number of epochs
                use_trained_model=FLAGS.use_trained_model,
                use_init_model=FLAGS.use_init_model
            )
        else:
            print('\n\tTesting Mode')
            model.custom_test(
                testing_samples_dir=FLAGS.testdir + '/*.jpg'
            )


if __name__ == '__main__':
    # with tf.device('/cpu:0'):
    tf.app.run()


