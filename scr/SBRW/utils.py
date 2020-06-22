import argparse
import os
import scipy.sparse as sps
from random import random
from scipy.sparse.linalg import norm, inv
from sklearn.metrics import f1_score,roc_auc_score

def get_arguments():
    parser = argparse.ArgumentParser(description= 'This is the code to run the SBRW sign prediction.')

    ################################################################################
    #                               general parameters
    ################################################################################

    parser.add_argument('--iterative_or_closed', type=str,
                        help='solve with iterative method or close form solution',
                        default='closed')
    
    parser.add_argument('--data_name', type=str, required=True,
                        help='data file name that then has _train, _val, _test')

    parser.add_argument('--val_or_test', type=str,
                        help='when running do we use val or test',
                        default='test')

    parser.add_argument('-c', '--restart_prob', type=float,
                        help='the restart probability for the random walk',
                        default=0.85)

    parser.add_argument('--iterative_convergence_threshold', type=float,
                        help='the threshold used when looking at the difference between iterations',
                        default=0.01)

    parser.add_argument('--omega', type=float,
                        help='real links have omega times the importance as projected links',
                        default=2)

    parser.add_argument('--delta_neg', type=float, required=True,
                        help='this is the threshold for which negative projected links to include'\
                        'Defaults: bonanza = X, senate = -10, house = -100')
    
    #Defaults: bonanza = X, senate = -10, house = -100
    parser.add_argument('--delta_pos', type=float, required=True,
                        help='this is the threshold for which positive projected links to include'\
                        'Defaults: bonanza = X, senate = 10, house = 50')
    
    ################################################################################
    args = parser.parse_args()
    print(args)

    args_dict = {}
    for arg in vars(args):
        args_dict[arg] = getattr(args, arg)
    return args_dict

def load_data(file_name):
    with open(file_name) as f:
        num_b, num_s, num_e = [int(value) for value in f.readline().split('\t')]
        num_total = num_s + num_b
        #will be num_s then num_b
        map_b = {i:i+num_s for i in range(num_b)}

        links = []
        signs =[]
        for l in f:
            l = [int(val) for val in l.split('\t')]
            b,s = map_b[l[0]], l[1]
            links.append((b,s))
            signs.append(float(l[2]))

    return num_b, num_s, links, signs, map_b


def get_onemode_projection_links(args):
    def load(saved_file):
        neg_threshold = args['delta_neg']
        pos_threshold = args['delta_pos']
        with open(saved_file) as f:    
            num_proj, num_proj_e = [int(val) for val in f.readline().split('\t')]
            links_proj = []
            signs_proj = []
            for l in f:
                i,j,r = [int(val) for val in l.split('\t')]
                if r > 0:
                    if r < pos_threshold:
                        continue #because it did not have enough support
                else:
                    if r > neg_threshold:
                        continue #because it did not have enough support
                links_proj.append((i,j))
                signs_proj.append(r)
        return links_proj, signs_proj

    def create(saved_Ps_file, saved_Pb_file):
        #this was quickly added here from another file, can clean later
        proj_s_datafile = saved_Ps_file
        proj_b_datafile = saved_Pb_file

        with open(args['data_name'] + 'training.txt') as f:
            num_b, num_s, num_e = [int(val) for val in f.readline().split('\t')]
            links = []
            bp_neighs = [set() for _ in range(num_b)]
            bn_neighs = [set() for _ in range(num_b)]
            sp_neighs = [set() for _ in range(num_s)]
            sn_neighs = [set() for _ in range(num_s)]
            for l in f:
                b,s,r = [int(val) for val in l.split('\t')]
                links.append(((b,s),r))
                if r == 1:
                    bp_neighs[b].add(s)
                    sp_neighs[s].add(b)
                else:
                    bn_neighs[b].add(s)
                    sn_neighs[s].add(b)

            b_proj_links = {}
            #for every buyer, iterate over their sellers
            for b in range(num_b):
                #if b % 1000 == 0:
                #    print('{} of {}'.format(b,num_b))
                b_val = {}
                for sp in bp_neighs[b]:
                    for bp in sp_neighs[sp]:
                        if bp in b_val:
                            b_val[bp] += 1 #since its ++
                        else:
                            b_val[bp] = 1
                    for bn in sn_neighs[sp]:
                        if bn in b_val:
                            b_val[bn] -= 1 #since its +-
                        else:
                            b_val[bn] = -1
                for sn in bn_neighs[b]:
                    for bp in sp_neighs[sn]:
                        if bp in b_val:
                            b_val[bp] -= 1#since its -+
                        else:
                            b_val[bp] = -1
                    for bn in sn_neighs[sn]:
                        if bn in b_val:
                            b_val[bn] += 1#since its --
                        else:
                            b_val[bn] = 1
                #we now see who b will be connected to
                for b_,val in b_val.items():
                    link = tuple(sorted([b,b_]))
                    if (val > 0) or (val < 0):
                        b_proj_links[link] = val
                    #else if it was zero we ignore it
            #we now have all the b_proj_links

            links_Pb, signs_Pb = [], []
            with open(proj_b_datafile, 'w') as f:
                f.write('{}\t{}\n'.format(num_b,len(b_proj_links)))
                lines = []
                negs = 0
                for (b1,b2),val in b_proj_links.items():
                    links_Pb.append((b1, b2))
                    signs_Pb.append(val)
                    lines.append('{}\t{}\t{}'.format(b1,b2,val))
                    if val < 0:
                        negs += 1
                f.write('\n'.join(lines))

            s_proj_links = {}
            #for every buyer, iterate over their sellers
            for s in range(num_s):
                #if s % 1000 == 0:
                #    print('{} of {}'.format(s,num_s))
                s_val = {}
                for bp in sp_neighs[s]:
                    for sp in bp_neighs[bp]:
                        if sp in s_val:
                            s_val[sp] += 1 #since its ++
                        else:
                            s_val[sp] = 1
                    for sn in bn_neighs[bp]:
                        if sn in s_val:
                            s_val[sn] -= 1 #since its +-
                        else:
                            s_val[sn] = -1
                for bn in sn_neighs[s]:
                    for sp in bp_neighs[bn]:
                        if sp in s_val:
                            s_val[sp] -= 1#since its -+
                        else:
                            s_val[sp] = -1
                    for sn in bn_neighs[bn]:
                        if sn in s_val:
                            s_val[sn] += 1#since its --
                        else:
                            s_val[sn] = 1
                #we now see who b will be connected to
                for s_,val in s_val.items():
                    link = tuple(sorted([s,s_]))
                    if (val > 0) or (val < 0):
                        s_proj_links[link] = val
                    #else if it was zero we ignore it
            #we now have all the b_proj_links

            links_Ps, signs_Ps = [], []
            with open(proj_s_datafile, 'w') as f:
                f.write('{}\t{}\n'.format(num_s,len(s_proj_links)))
                lines = []
                negs = 0
                for (s1,s2),val in s_proj_links.items():
                    links_Ps.append((s1,s2))
                    signs_Ps.append(val)
                    lines.append('{}\t{}\t{}'.format(s1,s2,val))
                    if val < 0:
                        negs += 1
                f.write('\n'.join(lines))
        return links_Ps, signs_Ps, links_Pb, signs_Pb


    #Do we have the Ps and Pb files
    saved_Ps_file = '{}_proj_{}.txt'.format(args['data_name'], 's')
    saved_Pb_file = '{}_proj_{}.txt'.format(args['data_name'], 'b')
    if os.path.exists(saved_Ps_file) and os.path.exists(saved_Pb_file):
        #we can load the precomputed file
        print('Loading Ps and Pb...')
        links_Ps, signs_Ps = load(saved_Ps_file)
        links_Pb, signs_Pb = load(saved_Pb_file)
    else:
        #create the precomputed file and use it
        print('Creating Ps and Pb...')
        links_Ps, signs_Ps, links_Pb, signs_Pb = create(saved_Ps_file, saved_Pb_file)

    return links_Ps, signs_Ps, links_Pb, signs_Pb

def calculate_SBRW_transition_matrix(num_b, num_s, links_tr, signs_tr,
                                     links_Ps, signs_Ps, links_Pb, signs_Pb, omega):
    #calculate the degrees for each
    num_total = num_b + num_s
    d = [0.0] * num_total
    d_proj = [0.0] * num_total 
    for (i,j) in links_tr:
        d[i] += 1.0
        d[j] += 1.0

    for (i,j),r in zip(links_Pb,signs_Pb):
        d_proj[i] += abs(r)
        d_proj[j] += abs(r)

    for (i,j),r in zip(links_Ps,signs_Ps):
        d_proj[i] += abs(r)
        d_proj[j] += abs(r)

    # we now include the normalized versions
    Adok = sps.dok_matrix((num_total, num_total))

    for (i,j),r in zip(links_tr,signs_tr):
        Adok[i,j] = omega*(r/d[i])
        Adok[j,i] = omega*(r/d[j])

    for (i,j),r in zip(links_Pb,signs_Pb):
        Adok[i,j] = r/d_proj[i]
        Adok[j,i] = r/d_proj[j]

    for (i,j),r in zip(links_Ps,signs_Ps):
        Adok[i,j] = r/d_proj[i]
        Adok[j,i] = r/d_proj[j]

    #now for the final normalize to make a transition matrix   
    T = Adok.asformat('lil') 

    #now row normalize
    for i in range(num_total):
        temp_sum = abs(T[i,:]).sum()
        if temp_sum > 0:
            T[i,:] = T[i,:] / temp_sum

    T = T.asformat('csr') 
    return T


def evaluate(RR, neg_percent_tr, links_te, signs_te):
    preds = []
    guesses = 0
    for (i,j),r in zip(links_te,signs_te):
        if RR[i,j] == 0:
            #random guess
            guesses += 1
            if random() < neg_percent_tr:
                preds.append(-1)
            else:
                preds.append(1)
        else:
            if RR[i,j] > 0:
                preds.append(1)
            else:
                preds.append(-1)

    #calculate how well it performed
    auc = roc_auc_score(signs_te, preds)
    f1 = f1_score(signs_te, preds)
    print('AUC:{}\tF1:{}'.format(auc,f1))


def run_iterative(T, I, neg_percent_tr, links_te, signs_te, c, convergence_threshold):    
    R = T.copy().asformat('csc') #intialize with T
    R_ = T.copy().asformat('csc') #itialize with T
    norm2 = 999999999
    it = 0
    while norm2 > convergence_threshold:
        if it % 2 == 0:
            R_ = c*(T.dot(R)) + (1-c)*I
        else:
            R = c*(T.dot(R_)) + (1-c)*I

        norm2 = norm(R-R_)
        print('Iteration {} and difference {}'.format(it,norm2))
        it += 1

        #uncomment to see progress while converging
        #if it % 2 == 0:
        #    evaluate(R.copy().asformat('dok'), neg_percent_tr, links_te, signs_te)
        #else:
        #    evaluate(R_.copy().asformat('dok'), neg_percent_tr, links_te, signs_te)
            
    #get final evaluation
    #if we quit when it = 2 that means it = 1 was the last to execute
    #and so R was the last result
    if it % 2 == 0:
        evaluate(R.copy().asformat('dok'), neg_percent_tr, links_te, signs_te)
    else:
        evaluate(R_.copy().asformat('dok'), neg_percent_tr, links_te, signs_te)


def run_closed(T, I, neg_percent_tr, links_te, signs_te, c):
    T = T.asformat('csc')
    I = I.asformat('csc')
    RR = (1-c)*inv(I-c*T)
    evaluate(RR, neg_percent_tr, links_te, signs_te)
