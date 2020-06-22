from math import ceil
from random import shuffle
from sklearn.metrics import roc_auc_score, f1_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import config
import scipy.sparse as sps
import sys
import torch

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_buyers, n_sellers, n_factors=5):
        # go back and change n_factors later to 10, then 20
        # constructor that initializes the attributes of the object
        super(MatrixFactorization, self).__init__()
        # create buyer embeddings
        self.buyer_factors = torch.nn.Embedding(n_buyers, n_factors)#, sparse=True)
        # create seller embeddings
        self.seller_factors = torch.nn.Embedding(n_sellers, n_factors)#, sparse=True)

    def forward(self, b, s):
        b = Variable(b, requires_grad = False)
        s = Variable(s, requires_grad = False)
        return (self.buyer_factors(b)*self.seller_factors(s)).sum(1).sum(1)

    def predict(self, b, s):
        return (self.buyer_factors(b)*self.seller_factors(s)).sum(1)

    def save_me(self,output_file):
        torch.save(self.state_dict(), output_file)


def run_matrix_factorization(args, num_buyers, num_sellers, links_tr, links_te, 
                             signs_tr, signs_te, extra_links, extra_signs):
    zeros = Variable(torch.zeros([args.minibatch_size]), requires_grad = False)
    ones = Variable(torch.FloatTensor([1]*args.minibatch_size), requires_grad = False)

    #below used on the final minibatch that might be of smaller size
    zeros_left_over = Variable(torch.zeros([len(links_tr) % args.minibatch_size]),requires_grad = False)
    ones_left_over = Variable(torch.FloatTensor([1]*(len(links_tr) % args.minibatch_size)), requires_grad = False)

    def square_hinge(real,pred):
        try:
            loss = torch.max(zeros,(ones-real*pred))**2
        except:
            loss = torch.max(zeros_left_over,(ones_left_over-real*pred))**2            
        return torch.mean(loss)            
            
    extra_links_tensor = torch.LongTensor(extra_links)
    extra_signs_tensor = torch.FloatTensor(extra_signs)
    extra_signs_tensor = extra_signs_tensor.unsqueeze(1)
    extra_tensor = TensorDataset(extra_links_tensor, extra_signs_tensor)
    extra_tensor_loader = DataLoader(extra_tensor, shuffle=True, batch_size=args.minibatch_size)
    extra_tensor_iterator = iter(extra_tensor_loader)
                    
    links_tr_tensor = torch.LongTensor(links_tr)
    signs_tr_tensor = torch.FloatTensor(signs_tr)
    signs_tr_tensor = signs_tr_tensor.unsqueeze(1)
    tr_tensor = TensorDataset(links_tr_tensor, signs_tr_tensor)
    tr_tensor_loader = DataLoader(tr_tensor, shuffle=True, batch_size=args.minibatch_size)
    tr_tensor_iterator = iter(tr_tensor_loader)
    
    b_te_tensor = torch.LongTensor([b for b,s in links_te])
    s_te_tensor = torch.LongTensor([s for b,s in links_te])
    signs_te_tensor = torch.FloatTensor(signs_te)
    
    model = MatrixFactorization(num_buyers, num_sellers, n_factors=args.dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,  weight_decay=args.reg) 
    num_minibatches_per_epoch = int(ceil(len(links_tr)/int(args.minibatch_size)))
    extra_num_minibatches_per_epoch = int(ceil(len(extra_links)/int(args.minibatch_size)))
    mod_balance = args.mod_balance
    alpha = args.alpha
    for it in range(args.num_epochs):
        model.train()
        print('epoch {} of {}'.format(it,args.num_epochs))
        for i in range(num_minibatches_per_epoch):
            if (i % mod_balance) != 0:
                #regular links
                try:
                    b_s, sign = next(tr_tensor_iterator)
                except:
                    tr_tensor_iterator = iter(tr_tensor_loader)
                    b_s, sign = next(tr_tensor_iterator)
                b = b_s[:,:1]
                s = b_s[:,1:2]
                prediction = model(b,s)
                loss = square_hinge(Variable(sign), prediction)

            else:
                #extra links
                try:
                    b_s, sign = next(extra_tensor_iterator)
                except:
                    extra_tensor_iterator = iter(extra_tensor_loader)
                    b_s, sign = next(extra_tensor_iterator)
                b = b_s[:,:1]
                s = b_s[:,1:2]
                prediction = model(b,s)
                loss = alpha * square_hinge(Variable(sign), prediction)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #printing progress
        if it % 5 == 0:
            model.eval()
            predicted = []
            for (b,s), sign in zip(links_te, signs_te):
                b = Variable(torch.LongTensor([int(b)]))
                s = Variable(torch.LongTensor([int(s)]))
                prediction = model.predict(b,s)
                if prediction.data[0] > 0:
                    predicted.append(1)
                else:
                    predicted.append(-1)

            auc = roc_auc_score(signs_te, predicted)
            f1 = f1_score(signs_te,predicted)
            print('epoch {}, auc {}, f1 {}'.format(it,auc,f1))
            sys.stdout.flush()

    #Done training the model, so we save it
    model.save_me(args.output_directory)

    model.eval()
    #can switch to tensor version
    #prediction = model.predict(b_te_tensor, s_te_tensor)
    #until then the below works
    predicted = []
    for (b,s), sign in zip(links_te, signs_te):
        b = Variable(torch.LongTensor([int(b)]))
        s = Variable(torch.LongTensor([int(s)]))
        prediction = model.predict(b,s)
        if prediction.data[0] > 0:
            predicted.append(1)
        else:
            predicted.append(-1)

    # calculate AUC and F1
    auc = roc_auc_score(signs_te, predicted)
    f1 = f1_score(signs_te,predicted)
    return auc,f1


def run(args):
    GRID_SEARCH_TUNING = args.tuning
    random_seed = args.random_seed
    torch.manual_seed(random_seed)

    training_datafile = args.file_dataset + 'training.txt'
    #if tuning with grid search then use validation set
    if GRID_SEARCH_TUNING:
        testing_datafile = args.file_dataset + 'validation.txt'
    else:
        testing_datafile = args.file_dataset + 'testing.txt'

    #get training data
    with open(training_datafile) as f:
        num_b, num_s, num_e_tr = [int(val) for val in f.readline().split('\t')]
        links_tr = []
        signs_tr = []
        for l in f:
            b,s,r = [int(val) for val in l.split('\t')]
            links_tr.append((b,s))
            signs_tr.append(r)
    #get testing/validation data (called testing here)
    with open(testing_datafile) as f:
        num_b, num_s, num_e_te = [int(val) for val in f.readline().split('\t')]
        links_te = []
        signs_te = []
        for l in f:
            b,s,r = [int(val) for val in l.split('\t')]
            links_te.append((b,s))
            signs_te.append(r)

    extra_links = []
    extra_signs = []

    extra_pos_training_datafile = args.extra_training + 'extra_pos_links_sorted_from_B_balance_theory.txt'
    extra_pos_num = args.extra_pos_num
    with open(extra_pos_training_datafile) as f:
        for i, l in enumerate(f):
            if i >= extra_pos_num:
                break
            b,s,r = [int(val) for val in l.split('\t')]
            extra_links.append((b,s))
            extra_signs.append(1)

    extra_neg_training_datafile = args.extra_training + 'extra_neg_links_sorted_from_B_balance_theory.txt'
    extra_neg_num = args.extra_neg_num
    with open(extra_neg_training_datafile) as f:
        for i, l in enumerate(f):
            if i >= extra_neg_num:
                break
            b,s,r = [int(val) for val in l.split('\t')]
            extra_links.append((b,s))
            extra_signs.append(-1)

    auc, f1 = run_matrix_factorization(args, num_b, num_s,
                                       links_tr, links_te,
                                       signs_tr, signs_te,
                                       extra_links, extra_signs)

    print('AUC:{}\tF1:{}'.format(auc, f1))


if __name__ == '__main__':
    args = config.args
    run(args)
