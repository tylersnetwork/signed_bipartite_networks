#load the file of type
#N M
#i j s
#create the Ep, En, Nbp, Nbn, Nsp, and Nsn structures as defined below

#Ep is set of positive edges of the form (i,x)
#En is set of negative edges of the form (i,x)
#only the (buyer,seller) are in this set since we are undirected anyway

#Nbp[i] = set of positive neighbors of buyer i
#Nbn[i] = set of negative neighbors of buyer i
#Nsp[x] = set of positive neighbors of seller x
#Nsn[x] = set of negative neighbors of seller x

import sys
filename = sys.argv[1]
with open(filename) as f:
    n1,n2,ne = [int(v) for v in f.readline().split('\t')]
    Ep = set()
    En = set()
    Nbp = [set() for i in range(n1)]
    Nbn = [set() for i in range(n1)]
    Nsp = [set() for x in range(n2)]
    Nsn = [set() for x in range(n2)]

    for l in f:
        i,x,s = [int(v) for v in l.split('\t')]
        if s == 1:
            Ep.add((i,x))
            Nbp[i].add(x)
            Nsp[x].add(i)
        else:
            En.add((i,x))
            Nbn[i].add(x)
            Nsn[x].add(i)
        
#R results e.g.,
# b1 = 1 [++++]                                                                                                         = repeated 4x
# b2 = 2 [----]                                                                                                         = repeated 4x
# b3 = 3 [++--] same as [--++]               = buyer 2pos and buyer 2 neg, sellers each have 1 pos and 1 neg            = repeated 2x
# b4 = 4 [+-+-] same as [-+-+]               = both buyer and sellers have 1 pos and 1 neg                              = repeated 2x 
# b5 = 5 [+--+] same as [-++-]               = seller 2 pos and seller 2 neg, buyers each have 1 pos and 1 neg          = repeated 2x
# b6 = 6 [+---] same as all single pos 3 neg = one buyer has +- and one buyer --, one seller has +- and one seller --   = only 1x
# b7 = 7 [+++-] same as all single neg 3 pos = one buyer has +- and one buyer ++, one seller has +- and one seller ++   = only 1x

mapper = {1:'++++', 2:'----', 3:'++--', 4:'+-+-', 5:'+--+', 6:'+---', 7:'+++-'}
R = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}

for counter,(i,j) in enumerate(Ep):
    if counter % 10000 == 0:
        print('at counter {} of {}'.format(counter, len(Ep)))
        sys.stdout.flush()
        
    for jp in Nbp[i]:
        if jp == j:
            continue
        for ip in Nsp[j]:
            if ip == i:
                continue
            if (ip,jp) in Ep:
                R[1] += 1 #gets counted x4 here #['++++']
            elif (ip,jp) in En:
                R[7] += 1 #gets counted only 1x # #['+++-']

        #second seller is through +
        for ip in Nsn[j]:
            if ip == i:
                continue
            if (ip,jp) in En:
                R[3] += 1 #gets counted x2 here #['++--']

    for ip in Nsp[j]:
        if ip == i:
            continue
        #second buyer is through +
        for jp in Nbn[i]:
            if jp == j:
                continue
            if (ip,jp) in En:
                R[5] += 1 #gets counted x2 here #['+--+']

    for jp in Nbn[i]:
        if jp == j:
            continue
        for ip in Nsn[j]:
            if ip == i:
                continue
            if (ip,jp) in Ep:
                R[4] += 1 #gets counted 2x here #['+-+-']
            elif (ip,jp) in En:
                R[6] += 1 #gets counted 1x #['+---']
            
for counter, (i,j) in enumerate(En):
    if counter % 10000 == 0:
        print('at counter {} of {}'.format(counter, len(En)))
        sys.stdout.flush()

    for jp in Nbn[i]:
        if jp == j:
            continue
        for ip in Nsn[j]:
            if ip == i:
                continue
            if (ip,jp) in En:
                R[2] += 1 #gets counted 4x #['----']

total = 0                
for k,v in sorted(R.items()):
    total += v

                
neg = len(En) / (len(En) + len(Ep))
pos = 1 - neg

mapper = {1:'++++', 2:'----', 3:'++--', 4:'+-+-', 5:'+--+', 6:'+---', 7:'+++-'}
pppp = pos**4
nnnn = neg**4
ppnn = (pos*pos*neg*neg) * 2 # two ways of this, a buyer has 2 pos, a buyer has 2 neg, both sellers have 1 pos/neg each
pnpn = (pos*pos*neg*neg) * 2 # two ways of this, sellers and buyers each have 1 pos/neg each
pnnp = (pos*pos*neg*neg) * 2 # two ways of this, a seller has 2 pos, a seller has 2 neg, both buyers have 1 pos/neg each
pnnn = (pos*neg*neg*neg) * 4 # 4 ways to select this one neg edge
pppn = (pos*pos*pos*neg) * 4 # 4 ways to select this one neg edge
expected_map = {1:pppp, 2:nnnn, 3:ppnn, 4:pnpn, 5:pnnp, 6:pnnn, 7:pppn}

t = 0
for k,v in expected_map.items():
    t += v

assert( abs(t - 1) < 0.001)

#suprise according to leskovec 2010
#s(Ti) = (Ti - E[Ti]) / sqrt( total *prioriProb(Ti) ( 1 - prioriProb(Ti)))
from math import sqrt
import scipy.stats as st

for i in range(1,len(mapper)+1):
    real = R[i]
    expected = expected_map[i] * total
    expected_prob = expected_map[i]
    try:
        surprise = (real - expected) / sqrt(total * expected_prob * (1-expected_prob))
    except:
        surprise = 'N/A'
    #pvalue = st.norm.cdf(surprise)
    print('type {} (count, real_perc, expected, expected_prob, surprise):\t{}\t{}\t{}\t{}\t{}'.format(
        mapper[i], real, real/total, expected, expected_prob, surprise))#, pvalue))


