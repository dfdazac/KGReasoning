# -*- coding: utf-8 -*-

import numpy as np
import random
import torch
import time

from typing import List, Tuple


def list2tuple(l):
    return tuple(list2tuple(x) if type(x)==list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)


flatten = lambda l: sum(map(flatten, l), []) if isinstance(l, tuple) else [l]


def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())


def set_global_seed(seed: int, is_deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return


def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return


def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries


def flatten_structure(query_structure):
    if type(query_structure) == str:
        return [query_structure]

    flat_structure = []
    for element in query_structure:
        flat_structure.extend(flatten_structure(element))

    return flat_structure


def query_to_atoms(query_structure, flat_ids):
    flat_structure = flatten_structure(query_structure)
    batch_size, query_length = flat_ids.shape
    assert len(flat_structure) == query_length

    query_triples = []
    variable = 0
    previous = flat_ids[:, 0]

    for i in range(1, query_length):
        if flat_structure[i] == 'r':
            variable -= 1
        elif flat_structure[i] == 'e':
            previous = flat_ids[:, i]
            variable += 1
            continue

        triples = torch.empty(batch_size, 3, device=flat_ids.device, dtype=torch.long)
        triples[:, 0] = previous
        triples[:, 1] = flat_ids[:, i]
        triples[:, 2] = variable

        query_triples.append(triples)
        previous = variable

    atoms = torch.stack(query_triples, dim=1)
    num_variables = variable * -1

    return atoms, num_variables


def make_batches(size: int, batch_size: int) -> List[Tuple[int, int]]:
    max_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, max_batch)]
    return res
