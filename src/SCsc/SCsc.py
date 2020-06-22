#Example of how to run this file, first is the input graph, second is the file to store the features of this training data
#python3 SCsc.py training_dataset.txt testing_dataset.txt

import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

#load the file of the following format
#Nb Ns M
#i j s

#create the Ep, En, Nbp, Nbn, Nsp, and Nsn structures as defined below

#Ep is set of positive edges of the form (i,x)
#En is set of negative edges of the form (i,x)
#only the (buyer,seller) are in this set since we are undirected anyway

#we would have Ep train/test and En train/test
#but also just E train (has both pos/neg)

#only allow the neighbors to be constructed from the training data excluding testing links
#Nbp[i] = set of positive neighbors of buyer i
#Nbn[i] = set of negative neighbors of buyer i
#Nsp[x] = set of positive neighbors of seller x
#Nsn[x] = set of negative neighbors of seller x

training_datafile = sys.argv[1]
testing_datafile = sys.argv[2]

links = []
signs = []
with open(training_datafile) as f:
    num_b, num_s, num_e = [int(val) for val in f.readline().split('\t')]
    for l in f:
        b,s,r = [int(val) for val in l.split('\t')]
        links.append((b,s))
        signs.append(r)

test_links = []
test_signs = []
with open(testing_datafile) as f:
    num_b, num_s, num_e = [int(val) for val in f.readline().split('\t')]
    for l in f:
        b,s,r = [int(val) for val in l.split('\t')]
        test_links.append((b,s))
        test_signs.append(r)

Ep = set()
En = set()
Nbp = [set() for i in range(num_b)]
Nbn = [set() for i in range(num_b)]
Nsp = [set() for x in range(num_s)]
Nsn = [set() for x in range(num_s)]

for (b,s),r in zip(links, signs):
    if r == 1:
        Ep.add((b,s))
        Nbp[b].add(s)
        Nsp[s].add(b)
    else:
        En.add((b,s))
        Nbn[b].add(s)
        Nsn[s].add(b)

#################################################################################
#Get the cycle features
#################################################################################
# for an edge (i,x) potentially +++, ++-, ..., --- so 8 combinations undirected
# of the format i--> y --> j --> x for the three edges

mapper = {0:'+++', 1:'++-', 2:'+-+', 3:'-++', 4:'+--', 5:'-+-', 6:'--+', 7:'---'}
features = []

###########################################
#iterate over each training edge in the network
for counter,(i,j) in enumerate(links):
    if counter % 10000 == 0:
        print('at counter {} of {} training'.format(counter, len(links)))

    temp = [0.0]*8
    for jp in Nbp[i]:
        if jp == j:
            continue
        for ip in Nsp[j]:
            if ip == i:
                continue
            if (ip,jp) in Ep:
                temp[0] += 1 #['+++']
            elif (ip,jp) in En:
                temp[2] += 1 #['+-+']

        #second seller is through +
        for ip in Nsn[j]:
            if ip == i:
                continue
            if (ip,jp) in Ep:
                temp[1] += 1 #['++-']
            elif (ip,jp) in En:
                temp[4] += 1 #['+--']

    for jp in Nbn[i]:
        if jp == j:
            continue
        for ip in Nsn[j]:
            if ip == i:
                continue
            if (ip,jp) in Ep:
                temp[5] += 1 #['-+-']
            elif (ip,jp) in En:
                temp[7] += 1 #['---']
        for ip in Nsp[j]:
            if ip == i:
                continue
            if (ip,jp) in Ep:
                temp[3] += 1 #['-++']
            elif (ip,jp) in En:
                temp[6] += 1 #['--+']

    features.append(temp)

###########################################
test_features = []
#iterate over each test edge in the network
for counter,(i,j) in enumerate(test_links):
    if counter % 10000 == 0:
        print('at counter {} of {} testing'.format(counter, len(test_links)))

    temp = [0.0]*8
    for jp in Nbp[i]:
        if jp == j:
            continue
        for ip in Nsp[j]:
            if ip == i:
                continue
            if (ip,jp) in Ep:
                temp[0] += 1 #['+++']
            elif (ip,jp) in En:
                temp[2] += 1 #['+-+']

        #second seller is through +
        for ip in Nsn[j]:
            if ip == i:
                continue
            if (ip,jp) in Ep:
                temp[1] += 1 #['++-']
            elif (ip,jp) in En:
                temp[4] += 1 #['+--']

    for jp in Nbn[i]:
        if jp == j:
            continue
        for ip in Nsn[j]:
            if ip == i:
                continue
            if (ip,jp) in Ep:
                temp[5] += 1 #['-+-']
            elif (ip,jp) in En:
                temp[7] += 1 #['---']
        for ip in Nsp[j]:
            if ip == i:
                continue
            if (ip,jp) in Ep:
                temp[3] += 1 #['-++']
            elif (ip,jp) in En:
                temp[6] += 1 #['--+']

    test_features.append(temp)
###########################################

#features for each link are the 8 signed catepillars, i's Nbp and Nbn, x's Nsp and Nsn = 12 features
#write them to file
with open(training_datafile[:-4].split('/')[-1]  + '_features_SCsc.txt','w') as f:
    lines = []
    for (i,x),temp in zip(links,features):
        output = [str(i),str(x)]
        output.extend([str(v) for v in temp])
        lines.append(','.join(output))
    f.write('\n'.join(lines))

with open(testing_datafile[:-4].split('/')[-1]  + '_features_SCsc.txt','w') as f:
    lines = []
    for (i,x),temp in zip(test_links,test_features):
        output = [str(i),str(x)]
        output.extend([str(v) for v in temp])
        lines.append(','.join(output))
    f.write('\n'.join(lines))

train_feats = features
train_labels = signs

test_feats = test_features
test_labels = test_signs

######################################
#Do the predictions and training
######################################

lr = LogisticRegression(class_weight='balanced',C=1)
lr.fit(train_feats, train_labels)
test_pred = lr.predict(test_feats)
auc = roc_auc_score(test_labels, test_pred)
f1 = f1_score(test_labels, test_pred)
print('SCsc: AUC:{}\tF1:{}'.format(auc,f1))
