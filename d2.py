# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from typing import Callable


def query_1p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor]) -> Tensor:
    s_emb = entity_embeddings(queries[:, 0])
    p_emb = predicate_embeddings(queries[:, 1])
    candidates_emb = entity_embeddings.weight
    res = scoring_function(s_emb, p_emb, candidates_emb)
    return res


def query_2p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int = 10) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])
    p1_emb = predicate_embeddings(queries[:, 1])
    p2_emb = predicate_embeddings(queries[:, 2])

    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]
    k_ = min(k, nb_entities)

    batch_size = s_emb.shape[0]
    emb_size = s_emb.shape[1]

    # [B, N]
    atom1_scores_2d = scoring_function(s_emb, p1_emb, candidates_emb)

    # [B, K], [B, K]
    atom1_k_scores_2d, atom1_k_indices = torch.topk(atom1_scores_2d, k=k_, dim=1)

    # [B, K, E]
    x1_k_emb_3d = entity_embeddings(atom1_k_indices)
    assert emb_size == x1_k_emb_3d.shape[2]

    # "[B*K, E]"
    x1_k_emb_2d = x1_k_emb_3d.reshape(-1, emb_size)

    # [B*K, E]
    p2_k_emb_2d = p2_emb.reshape(batch_size, 1, emb_size).repeat(1, k_, 1).reshape(-1, emb_size).reshape(-1, emb_size)

    # [B*K, N]
    atom2_scores_2d = scoring_function(x1_k_emb_2d, p2_k_emb_2d, candidates_emb)

    # [B, K, N]
    atom1_k_scores_2d = atom1_k_scores_2d.reshape(-1, 1).repeat(1, nb_entities).reshape(-1, k_, nb_entities)
    atom2_scores_2d = atom2_scores_2d.reshape(-1, k_, nb_entities)

    res = torch.minimum(atom1_k_scores_2d, atom2_scores_2d)
    res, _ = torch.max(res, dim=1)
    return res


def query_3p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int = 10) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])
    p1_emb = predicate_embeddings(queries[:, 1])
    p2_emb = predicate_embeddings(queries[:, 2])
    p3_emb = predicate_embeddings(queries[:, 3])

    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]
    k_ = min(k, nb_entities)

    batch_size = s_emb.shape[0]
    emb_size = s_emb.shape[1]

    # [B, N]
    atom1_scores = scoring_function(s_emb, p1_emb, candidates_emb)

    # [B, K], [B, K]
    atom1_k_scores_2d, atom1_k_indices = torch.topk(atom1_scores, k=k_, dim=1)

    # [B, K, E]
    x1_k_emb_3d = entity_embeddings(atom1_k_indices)
    assert emb_size == x1_k_emb_3d.shape[2]

    # "[B*K, E]"
    x1_k_emb_2d = x1_k_emb_3d.reshape(-1, emb_size)

    # [B*K, E]
    p2_k_emb_2d = p2_emb.reshape(batch_size, 1, emb_size).repeat(1, k_, 1).reshape(-1, emb_size).reshape(-1, emb_size)

    # [B*K, N]
    atom2_scores_2d = scoring_function(x1_k_emb_2d, p2_k_emb_2d, candidates_emb)

    # [B*K, K], [B*K, K]
    atom2_k_scores_2d, atom2_k_indices = torch.topk(atom2_scores_2d, k=k_, dim=1)

    # [B*K, K, E]
    x2_k_emb_3d = entity_embeddings(atom2_k_indices)
    assert emb_size == x2_k_emb_3d.shape[2]

    # "[B*K*K, E]"
    x2_k_emb_2d = x2_k_emb_3d.reshape(-1, emb_size)
    k2_ = x2_k_emb_2d.shape[0]

    # [B*K*K, E]
    p3_k_emb_2d = p2_emb.reshape(batch_size, 1, emb_size).repeat(1, k2_, 1).reshape(-1, emb_size).reshape(-1, emb_size)

    # [B*K*K, N]
    atom3_scores_2d = scoring_function(x2_k_emb_2d, p3_k_emb_2d, candidates_emb)

    # [B, K*K, N]
    atom1_k_scores_2d = atom1_k_scores_2d.reshape(-1, k_, 1).repeat(1, k_, nb_entities).reshape(-1, k2_, nb_entities)
    atom2_k_scores_2d = atom2_scores_2d.reshape(-1, k2_, 1).repeat(-1, 1, nb_entities).reshape(-1, k2_, nb_entities)
    atom3_scores_2d = atom3_scores_2d.reshape(-1, k2_, nb_entities)

    res = torch.minimum(atom1_k_scores_2d, atom2_scores_2d)
    res = torch.minimum(atom2_k_scores_2d, atom3_scores_2d)

    res, _ = torch.max(res, dim=1)
    return res