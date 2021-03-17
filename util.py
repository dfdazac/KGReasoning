from typing import List, Tuple
import numpy as np
import random
import torch
import time



Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

query_name_dict = {('e',('r',)): '1p',
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                }
                


def list2tuple(l):
    return tuple(list2tuple(x) if type(x)==list else x for x in l)

def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)

flatten=lambda l: sum(map(flatten, l),[]) if isinstance(l,tuple) else [l]

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

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

def make_batches(size: int, batch_size: int) -> List[Tuple[int, int]]:
    max_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, max_batch)]
    return res


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



def create_instructions(chains):
    instructions = []
    try:

        prev_start = None
        prev_end = None

        path_stack = []
        start_flag = True
        for chain_ind, chain in enumerate(chains):
            if start_flag:
                prev_end = chain[-1]
                start_flag = False
                continue


            if prev_end == chain[0]:
                instructions.append(f"hop_{chain_ind-1}_{chain_ind}")
                prev_end = chain[-1]
                prev_start = chain[0]

            elif prev_end == chain[-1]:

                prev_start = chain[0]
                prev_end = chain[-1]

                instructions.append(f"intersect_{chain_ind-1}_{chain_ind}")
            else:
                path_stack.append(([prev_start, prev_end],chain_ind-1))
                prev_start = chain[0]
                prev_end = chain[-1]
                start_flag = False
                continue

            if len(path_stack) > 0:

                path_prev_start = path_stack[-1][0][0]
                path_prev_end = path_stack[-1][0][-1]

                if path_prev_end == chain[-1]:

                    prev_start = chain[0]
                    prev_end = chain[-1]

                    instructions.append(f"intersect_{path_stack[-1][1]}_{chain_ind}")
                    path_stack.pop()
                    continue

        ans = []
        for inst in instructions:
            if ans:

                if 'inter' in inst and ('inter' in ans[-1]):
                        last_ind = inst.split("_")[-1]
                        ans[-1] = ans[-1]+f"_{last_ind}"
                else:
                    ans.append(inst)

            else:
                ans.append(inst)

        instructions = ans

    except RuntimeError as e:
        print(e)
        return instructions
    return instructions
