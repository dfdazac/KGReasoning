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
    command = f'PYTHONPATH=. python3 main.py --do_train --do_valid --do_test --data_path data/{c["d"]} ' \
              f'-n 1 -b {c["b"]} -d {c["k"]} -lr {c["lr"]} --disable_warmup --max_steps 100000 --cpu_num 0 --geo cqd ' \
              f'--valid_steps 500 --tasks "1p" --print_on_screen --test_batch_size 1000 --optimizer "Adagrad" ' \
              f'--reg_weight {c["n3"]} --log_steps 500 --cuda --no-save'
    return command


def to_logfile(c, path):
    outfile = "{}/cqd_beaker_v1.{}.log".format(path, summary(c).replace("/", "_"))
    return outfile


def main(argv):
    hyp_space = dict(
        k=[1000],
        b=[50, 100, 500, 1000],
        n3=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
        lr=[0.1],
        data=['FB15k-237-betae', 'FB15k-237-q2b', 'FB15k-betae', 'FB15k-q2b', 'NELL-betae', 'NELL-q2b']
    )

    configurations = list(cartesian_product(hyp_space))

    path = 'logs/cqd_beaker_v1'
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
#$ -l h_rt=48:00:00
#$ -l gpu=true

conda activate gpu

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/workspace/KGReasoning

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && sleep 30 && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
