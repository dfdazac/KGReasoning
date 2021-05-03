#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import pandas as pd

import argparse

from typing import Dict, Tuple

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def path_to_key(path: str) -> str:
    file_name = os.path.basename(path)
    key = os.path.splitext(file_name)[0]
    res = key.split('.')[1]
    return res


def key_to_dict(key: str) -> Dict[str, str]:
    res = {}
    for item in key.split('_'):
        k, v = item.split('=')
        res[k] = v
    return res


def path_to_results(path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    with open(path) as f:
        lines = f.readlines()

    res_dev, res_test = {}, {}

    for line in lines:
        if 'Valid average' in line:
            tokens = line.split()
            res_dev[tokens[5]] = float(tokens[9])

        if 'Test average' in line:
            tokens = line.split()
            res_test[tokens[5]] = float(tokens[9])

    return res_dev, res_test


def main(argv):
    parser = argparse.ArgumentParser(description='Parse results.')
    parser.add_argument('paths', metavar='path', type=str, nargs='+')
    parser.add_argument('--mrr', action='store_true', default=False)

    args = parser.parse_args(argv)

    is_mrr = args.mrr

    metric = 'HITS3'
    if is_mrr is True:
        metric = 'MRR'

    show_metric = 'HITS3'

    key_to_path = {path_to_key(path): path for path in args.paths}

    data_values = set()
    query_values = set()

    for key in key_to_path:
        d = key_to_dict(key)

        if d['data'] not in data_values:
            data_values.add(d['data'])
        if d['q'] not in query_values:
            query_values.add(d['q'])

    data_values = sorted(data_values)
    query_values = sorted(query_values)

    query_values = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'inp', 'pin', 'pni']

    data_pd_values = []
    metric_pd_values = []
    query_pd_values = []
    result_pd_values = []

    has_query = set()

    for data in data_values:
        for q in query_values:

            best_res_dev = best_res_test = None

            for key in key_to_path:
                d = key_to_dict(key)

                if d['data'] == data and d['q'] == q:
                    path = key_to_path[key]

                    res_dev, res_test = path_to_results(path)

                    if len(res_dev) > 0:

                        if best_res_dev is None or res_dev[metric] > best_res_dev[metric]:
                            best_res_dev = res_dev
                            best_res_test = res_test

            if best_res_test is not None and len(best_res_test) > 0:
                for k, v in best_res_test.items():
                    has_query.add(q)

                    data_pd_values += [data]
                    query_pd_values += [q]
                    metric_pd_values += [k]
                    result_pd_values += [v]

    pd_dict = {
        'Data': data_pd_values,
        'Query': query_pd_values,
        'Metric': metric_pd_values,
        'Result': result_pd_values
    }

    df = pd.DataFrame(pd_dict)
    metric_df = df[df.Metric == show_metric]

    new_df = metric_df.set_index(['Data', 'Query']).drop(['Metric'], axis=1).unstack()

    query_values = [q for q in query_values if q in has_query]
    new_df = new_df[[('Result', q) for q in query_values]]

    print(new_df.to_latex())


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
