python3 get_extra_links.py ../../data/dataset_training.txt ../../data/dataset_testing.txt ../../data/dataset_validation.txt 1000 1000 extra_links/

python3 MFwBT.py --num_epochs 20 --minibatch_size 1 --file_dataset ../../data/dataset_ --dim 5 --learning_rate 0.05 --reg 0 --extra_training extra_links/dataset_ --extra_pos_num 1000 --extra_neg_num 1000 --mod_balance 5

Note the naming is assumed to be dataset_training.txt, _testing.txt, and _validation.txt for the split training, testing andvalidation sets.
