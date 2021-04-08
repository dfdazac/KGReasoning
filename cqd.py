# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch import optim, Tensor
import math

from util import query_to_atoms

from typing import Tuple


class N3(nn.Module):
    def __init__(self, weight: float):
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class CQD(nn.Module):
    def __init__(self, nentity: int, nrelation: int, rank: int,
                 init_size: float = 1e-3, reg_weight: float = 1e-2, test_batch_size: int = 1):
        super(CQD, self).__init__()

        self.rank = rank
        self.nentity = nentity
        self.nrelation = nrelation

        sizes = (nentity, nrelation)
        self.embeddings = nn.ModuleList([nn.Embedding(s, 2 * rank, sparse=True) for s in sizes[:2]])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.init_size = init_size
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.regularizer = N3(reg_weight)

        batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1)
        self.register_buffer('batch_entity_range', batch_entity_range)

    def split(self,
              lhs_emb: Tensor,
              rel_emb: Tensor,
              rhs_emb: Tensor):
        lhs = lhs_emb[..., :self.rank], lhs_emb[..., self.rank:]
        rel = rel_emb[..., :self.rank], rel_emb[..., self.rank:]
        rhs = rhs_emb[..., :self.rank], rhs_emb[..., self.rank:]
        return lhs, rel, rhs

    def loss(self, triples):
        (scores_o, scores_s), factors = self.score_candidates(triples)
        l_fit = self.loss_fn(scores_o, triples[:, 2]) + self.loss_fn(scores_s, triples[:, 0])
        l_reg = self.regularizer.forward(factors)
        return l_fit + l_reg

    def score_candidates(self, triples: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        lhs_emb = self.embeddings[0](triples[:, 0])
        rel_emb = self.embeddings[1](triples[:, 1])
        rhs_emb = self.embeddings[0](triples[:, 2])

        to_score = self.embeddings[0].weight
        scores_o, _ = self.score_o(lhs_emb, rel_emb, to_score)
        scores_s, _ = self.score_s(to_score, rel_emb, rhs_emb)

        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        factors = self.get_factors(lhs, rel, rhs)

        return (scores_o, scores_s), factors

    def score_o(self, lhs_emb, rel_emb, rhs_emb, return_factors=False):
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        score_1 = (lhs[0] * rel[0] - lhs[1] * rel[1]) @ rhs[0].transpose(-1, -2)
        score_2 = (lhs[1] * rel[0] + lhs[0] * rel[1]) @ rhs[1].transpose(-1, -2)
        factors = self.get_factors(lhs, rel, rhs) if return_factors else None
        return score_1 + score_2, factors

    def score_s(self, lhs_emb, rel_emb, rhs_emb, return_factors=False):
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        score_1 = (rhs[0] * rel[0] + rhs[1] * rel[1]) @ lhs[0].transpose(-1, -2)
        score_2 = (rhs[1] * rel[0] - rhs[0] * rel[1]) @ lhs[1].transpose(-1, -2)
        factors = self.get_factors(lhs, rel, rhs) if return_factors else None
        return score_1 + score_2, factors

    def get_factors(self, lhs, rel, rhs):
        factors = []
        for term in (lhs, rel, rhs):
            factors.append(torch.sqrt(term[0] ** 2 + term[1] ** 2))

        return factors

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
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
                h_emb_constants = self.embeddings[0](head)
                r_emb = self.embeddings[1](rel)

            if num_variables > 1:
                # var embedding for ID 0 is unused for ease of implementation
                var_embs = nn.Embedding((num_variables * batch_size) + 1,
                                        self.rank * 2)
                var_embs.weight.data *= self.init_size

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

                    scores, factors = self.score_o(h_emb.unsqueeze(-2),
                                                   r_emb.unsqueeze(-2),
                                                   t_emb.unsqueeze(-2),
                                                   return_factors=True)
                    t_norm = torch.prod(torch.sigmoid(scores), dim=1)
                    loss = -t_norm.mean() + self.regularizer.forward(factors)
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
                to_score = self.embeddings[0].weight

                scores, factors = self.score_o(h_emb, r_emb, to_score)
                scores = torch.sigmoid(scores)
                t_norm = torch.prod(scores, dim=1)

                all_scores.append(t_norm)

        scores = torch.cat(all_scores, dim=0)

        return None, scores, None, all_idxs
