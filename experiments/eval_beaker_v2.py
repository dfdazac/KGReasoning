#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs if k not in {'d'}])


def to_cmd(c, _path=None):
    s_normalize = '--cqd-normalize' if c["n"] else ''
    s_sigmoid = '--cqd-sigmoid' if c["s"] else ''
    command = f'PYTHONPATH=. python3 main.py --do_valid --do_test ' \
              f'--data_path data/{c["data"].replace("fb", "FB").replace("nell", "NELL")} ' \
              f'-n 1 -b 1000 -d 1000 -lr 0.1 ' \
              f'--disable_warmup --max_steps 1000 --cpu_num 0 --geo cqd --valid_steps 20  --tasks "{c["q"]}" ' \
              f'--print_on_screen --test_batch_size 1 --optimizer "Adagrad" --reg_weight 0.05 ' \
              f'--log_steps 5 --checkpoint_path models/{c["data"]} ' \
              f'--cqd d2 --cqd-t-norm {c["t"]} {s_normalize} {s_sigmoid} --cqd-k {c["k"]} --no-save --cuda'
    return command


def to_logfile(c, path):
    outfile = "{}/eval_beaker_v2.{}.log".format(path, summary(c).replace("/", "_"))
    return outfile


def main(argv):
    hyp_space = dict(
        q=['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u', 'up'],
        t=['min', 'prod'],
        s=[True, False],
        n=[True, False],
        k=[1, 2, 4, 8, 16, 32, 64],
        data=['fb15k-q2b', 'fb15k-betae', 'fb15k-237-q2b', 'fb15k-237-betae', 'nell-q2b', 'nell-betae']
    )

    configurations = list(cartesian_product(hyp_space))

    path = 'logs/eval_beaker_v2'
    is_rc = False

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/pminervi/'):
        is_rc = True
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if is_rc is True and os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Training finished' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    header = """#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -t 1-{}
#$ -l tmem=16G
#$ -l h_rt=2:00:00
#$ -l gpu=true

conda activate gpu

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/workspace/KGReasoning

""".format(nb_jobs)

    # header = ''

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && sleep 10 && {}'.format(job_id, command_line))
        # print(f'{command_line}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
