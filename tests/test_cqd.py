# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import Tensor

from cqd import CQD

import pytest


@pytest.mark.light
def test_complex_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    rs = np.random.RandomState(0)

    for _ in range(128):
        with torch.no_grad():

            model = CQD(nentity=nb_entities, nrelation=nb_predicates, rank=embedding_size, init_size=1.0)

            def score(rel: Tensor,
                      arg1: Tensor,
                      arg2: Tensor) -> Tensor:
                # [B, E]
                rel_real, rel_img = rel[:, :embedding_size], rel[:, embedding_size:]
                arg1_real, arg1_img = arg1[:, :embedding_size], arg1[:, embedding_size:]
                arg2_real, arg2_img = arg2[:, :embedding_size], arg2[:, embedding_size:]
                # [B] Tensor
                res = torch.sum(rel_real * arg1_real * arg2_real +
                                rel_real * arg1_img * arg2_img +
                                rel_img * arg1_real * arg2_img -
                                rel_img * arg1_img * arg2_real, 1)
                # [B] Tensor
                return res

            xs = torch.tensor(rs.randint(nb_entities, size=32), dtype=torch.long)
            xp = torch.tensor(rs.randint(nb_predicates, size=32), dtype=torch.long)
            xo = torch.tensor(rs.randint(nb_entities, size=32), dtype=torch.long)

            xs_emb = model.embeddings[0](xs)
            xp_emb = model.embeddings[1](xp)
            xo_emb = model.embeddings[0](xo)

            to_score = model.embeddings[0].weight
            scores_sp, _ = model.score_o(xs_emb, xp_emb, to_score)
            scores_po, _ = model.score_s(to_score, xp_emb, xo_emb)

            inf = score(xp_emb, xs_emb, xo_emb)

            inf = inf.cpu().numpy()
            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            for i in range(32):
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    # test_complex_v1()
    pytest.main([__file__])
