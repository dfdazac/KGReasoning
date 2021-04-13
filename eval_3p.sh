#!/bin/bash

PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-237-q2b -n 1 -b 1000 -d 1000 -lr 0.1 --disable_warmup \
 --max_steps 1000 --cpu_num 0 --geo cqd --valid_steps 20  --tasks "3p" --print_on_screen --test_batch_size 1000 \
  --optimizer "Adagrad" --reg_weight 0.05 --log_steps 5 \
  --checkpoint_path "logs/FB15k-237-q2b/1p/cqd/2021.04.07-20:45:36" $@
