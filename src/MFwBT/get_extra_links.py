from collections import Counter
import numpy as np
import scipy.sparse as sps
import sys
import os

training_datafile = sys.argv[1]
testing_datafile = sys.argv[2]
validation_datafile = sys.argv[3]
topk_neg_to_use = int(sys.argv[4])
topk_pos_to_use = int(sys.argv[5])
prefix = sys.argv[6]
if not os.path.exists:
    os.makedirs(prefix)

with open(training_datafile) as f:
    num_b, num_s, num_e = [int(value) for value in f.readline().split('\t')]
    print('num buyers and sellers:', num_b, num_s) 
    links = []
    signs =[]
    for l in f:
        l = [int(val) for val in l.split('\t')]
        links.append((l[0],l[1]))
        signs.append(l[2])
num_e_pos = signs.count(1)
num_e_neg = signs.count(-1)
print('num e_pos and e_neg: ', num_e_pos, num_e_neg)

B = sps.dok_matrix((num_b, num_s))
for (b,s),r in zip(links,signs):
    B[b,s] = float(r)

B = B.asformat('csc')
S = (B.dot(B.T)).dot(B).asformat('dok')
B = B.asformat('dok')

assumed_links = []
pos_with_link, neg_with_link = [], []
for (b,s),val in S.items():
    if ((b,s) not in B):
        if val > 0:
            pos_with_link.append((val,(b,s)))
            assumed_links.append('{}\t{}\t{}'.format(b,s,1))
        else:
            neg_with_link.append((val,(b,s)))
            assumed_links.append('{}\t{}\t{}'.format(b,s,-1))

print('total nonzero: ', len(S.keys()))
print('total not in B: ', len(neg_with_link) + len(pos_with_link))
print('total not in B neg: ', len(neg_with_link))

with open('{}extra_links_from_B_balance_theory.txt'.format(prefix),'w') as f:
    f.write('\n'.join(assumed_links))

neg_with_link.sort()
with open('{}extra_neg_links_sorted_from_B_balance_theory.txt'.format(prefix), 'w') as f:
    lines = ['{}\t{}\t{}'.format(b,s,-1) for val,(b,s) in neg_with_link]
    f.write('\n'.join(lines))

pos_with_link.sort(reverse=True)
with open('{}extra_pos_links_sorted_from_B_balance_theory.txt'.format(prefix), 'w') as f:
    lines = ['{}\t{}\t{}'.format(b,s,1) for val,(b,s) in pos_with_link]
    f.write('\n'.join(lines))
