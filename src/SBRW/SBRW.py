from utils import *

def run(args):
    num_b, num_s, links_tr, signs_tr, map_b = load_data(args['data_name'] + 'training.txt')

    if args['val_or_test'] == 'test':
        num_b, num_s, links_te, signs_te, map_b = load_data(args['data_name'] + 'testing.txt')
    elif args['val_or_test'] == 'val':
        num_b, num_s, links_te, signs_te, map_b = load_data(args['data_name'] + 'validation.txt')

    num_e_tr, num_e_te = len(links_tr), len(links_te)

    links_Ps, signs_Ps, links_Pb, signs_Pb = get_onemode_projection_links(args)
    T = calculate_SBRW_transition_matrix(num_b, num_s, links_tr, signs_tr,
                                             links_Ps, signs_Ps, links_Pb, signs_Pb,args['omega'])
    I = sps.eye(num_s + num_b)
    #used for the random guessing if needed
    neg_percent_tr = signs_tr.count(-1) / len(signs_tr)

    if args['iterative_or_closed'] == 'iterative':
        print('Running iterative method...')
        run_iterative(T, I, neg_percent_tr, links_te, signs_te,
                      args['restart_prob'], args['iterative_convergence_threshold'])
    elif args['iterative_or_closed'] == 'closed':
        print('Running closed form method...')
        run_closed(T, I, neg_percent_tr, links_te, signs_te, args['restart_prob'])
    else:
        raise Exception('Should be "iterative" or "closed" for the method to solve.')
    
if __name__ == '__main__':
    args = get_arguments()
    run(args)
