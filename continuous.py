# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch import optim, Tensor
import math

from util import query_to_atoms
from cqd import N3

from typing import Tuple, Dict, Any, Callable


def optimize(batch_queries_dict: Dict[Tuple, Tensor],
             batch_idxs_dict: Dict[Any, Any],
             embeddings: nn.ModuleList,
             rank: int,
             init_size: float,
             score_o: Callable,
             regularizer: N3) -> Tuple[Tensor, Tensor]:
    all_idxs = []
    all_scores = []

    for query_structure, queries in batch_queries_dict.items():
        batch_size = queries.shape[0]
        atoms, num_variables = query_to_atoms(query_structure, queries)
        all_idxs.extend(batch_idxs_dict[query_structure])

        target_mask = torch.sum(atoms == -num_variables, dim=-1) > 0

        # Offsets identify variables across different batches
        var_id_offsets = torch.arange(batch_size, device=atoms.device) * num_variables
        var_id_offsets = var_id_offsets.reshape(-1, 1, 1)

        # Replace negative variable IDs with valid identifiers
        vars_mask = atoms < 0
        atoms_offset_vars = -atoms + var_id_offsets
        atoms[vars_mask] = atoms_offset_vars[vars_mask]

        head, rel, tail = atoms[..., 0], atoms[..., 1], atoms[..., 2]
        head_vars_mask = vars_mask[..., 0]

        with torch.no_grad():
            h_emb_constants = embeddings[0](head)
            r_emb = embeddings[1](rel)

        if num_variables > 1:
            # var embedding for ID 0 is unused for ease of implementation
            var_embs = nn.Embedding((num_variables * batch_size) + 1, rank * 2)
            var_embs.weight.data *= init_size

            var_embs.to(atoms.device)
            optimizer = optim.Adam(var_embs.parameters(), lr=0.1)
            prev_loss_value = 1000
            loss_value = 999
            i = 0

            # CQD-CO optimization loop
            while i < 1000 and math.fabs(prev_loss_value - loss_value) > 1e-9:
                prev_loss_value = loss_value

                h_emb = h_emb_constants.clone()
                # Fill variable positions with optimizable embeddings
                h_emb[head_vars_mask] = var_embs(head[head_vars_mask])
                t_emb = var_embs(tail)

                scores, factors = score_o(h_emb.unsqueeze(-2),
                                          r_emb.unsqueeze(-2),
                                          t_emb.unsqueeze(-2),
                                          return_factors=True)
                t_norm = torch.prod(torch.sigmoid(scores), dim=1)
                loss = - t_norm.mean() + regularizer.forward(factors)
                loss_value = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1
        else:
            h_emb = h_emb_constants

        with torch.no_grad():
            # Select predicates involving target variable only
            target_mask = target_mask.unsqueeze(-1).expand_as(h_emb)
            emb_size = h_emb.shape[-1]
            h_emb = h_emb[target_mask].reshape(batch_size, -1, emb_size)
            r_emb = r_emb[target_mask].reshape(batch_size, -1, emb_size)
            to_score = embeddings[0].weight

            scores, factors = score_o(h_emb, r_emb, to_score)
            scores = torch.sigmoid(scores)
            t_norm = torch.prod(scores, dim=1)

            all_scores.append(t_norm)

    scores = torch.cat(all_scores, dim=0)

    return scores, all_idxs