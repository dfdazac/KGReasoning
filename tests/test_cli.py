# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import subprocess
import pytest


@pytest.mark.light
def test_cli_v1():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'

    cmd_str = 'python3 main.py --do_train --do_valid --do_test --data_path data/NELL-betae -n 1 -b 10 -d 100 ' \
              '-lr 0.1 --disable_warmup --max_steps 10 --cpu_num 0 --geo cqd --valid_steps 10 --tasks 1p ' \
              '--print_on_screen --test_batch_size 1000 --optimizer Adagrad --reg_weight 0.01 --log_steps 500 ' \
              '--no-save'

    cmd = cmd_str.split()

    sys.stdout = sys.stderr

    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    # print('AAA', out, 'BBB', err)
    lines = err.decode("utf-8").split("\n")

    sanity_check_flag_1 = False

    for line in lines:
        # print(line)
        if 'Test average MRR at step 9' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.016133, atol=1e-3, rtol=1e-3)
            sanity_check_flag_1 = True

    assert sanity_check_flag_1


if __name__ == '__main__':
    pytest.main([__file__])
    # test_cli_v1()
