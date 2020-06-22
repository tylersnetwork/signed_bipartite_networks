import argparse
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def path(p):
    return os.path.expanduser(p)

PATH = ''

parser = argparse.ArgumentParser(description='Arguments for Bipartite Signed Network Link Sign Prediction')

parser.add_argument('--random_seed', type=int, required=False, default=1234,
                    help='random seed that can be used to get reproducible results')
################################################################################
#          General parameters related to the optimization process
################################################################################
parser.add_argument('--learning_rate', type=float, required=False, default=0.01,
                    help='the learning rate for optimizing')
parser.add_argument('--tuning', type=str2bool, nargs='?', const=True, default=False,
                    help='are we doing a grid search tuning run? (T/F) or actually using the testing data')
parser.add_argument('--num_epochs', type=int, required=False, default=3,
                    help='the maximum number of epochs allowed in relation to the largest training dataset type during the optimization procedure')
parser.add_argument('--minibatch_size', type=int, required=False, default=100,
                    help='how many links we use for training at each time we update')
parser.add_argument('--alpha', type=float, required=False, default=0.1,
                    help='alpha: controls the contribution of the implicit link signs obtained from balance theory')
parser.add_argument('--reg', type=float, required=False, default=0.001,
                    help='reg: controls the regularization of the U and V matrices to prevent overfitting')

################################################################################
#                  Parameters related to the dataset files
################################################################################
parser.add_argument('--file_dataset', type=path, required=True,
                    help='dataset file name which will be parsed to get all needed files')
parser.add_argument('--output_directory', type=path, required=False, default='model_file',
                    help='the output directory name for the given model output and config file settings')
parser.add_argument('--extra_training', type=path, required=True,
                    help='the path to extra links')
parser.add_argument('--extra_pos_num', type=int, required=True, 
                    help='how many positive links to include from balance theory')
parser.add_argument('--extra_neg_num', type=int, required=True,
                    help='how many negative links to include from balance theory')
parser.add_argument('--mod_balance', type=int, required=True,
                    help='how many to mod for balance of the two types of links')
################################################################################
#          Parameters related to the matrix factorization model
################################################################################
parser.add_argument('--dim', type=int, required=False, default=10,
                    help='the dimension of the latent preference vector for each user')

args = parser.parse_args()
