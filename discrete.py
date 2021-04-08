# -*- coding: utf-8 -*-

import torch

from tqdm import tqdm

from util import make_batches


def top_k_selection(self,
                    chains,
                    chain_instructions,
                    graph_type,
                    candidates: int = 5,
                    t_norm: str = 'min',
                    batch_size: int = 1,
                    scores_normalize: int = 0):
    res = None

    if 'disj' in graph_type:
        objective = self.t_conorm
    else:
        objective = self.t_norm

    nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]

    scores = None

    batches = make_batches(nb_queries, batch_size)

    for batch in tqdm(batches):

        nb_branches = 1
        nb_ent = 0
        batch_scores = None
        candidate_cache = {}

        batch_size = batch[1] - batch[0]
        dnf_flag = False
        if 'disj' in graph_type:
            dnf_flag = True

        for inst_ind, inst in enumerate(chain_instructions):
            with torch.no_grad():
                if 'hop' in inst:

                    ind_1 = int(inst.split("_")[-2])
                    ind_2 = int(inst.split("_")[-1])

                    indices = [ind_1, ind_2]

                    if objective == self.t_conorm and dnf_flag:
                        objective = self.t_norm

                    last_hop = False
                    for hop_num, ind in enumerate(indices):
                        last_step = (inst_ind == len(chain_instructions) - 1) and last_hop

                        lhs, rel, rhs = chains[ind]

                        if lhs is not None:
                            lhs = lhs[batch[0]:batch[1]]
                        else:
                            # print("MTA BRAT")
                            batch_scores, lhs_3d = candidate_cache[f"lhs_{ind}"]
                            lhs = lhs_3d.view(-1, embedding_size)

                        rel = rel[batch[0]:batch[1]]
                        rel = rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                        rel = rel.view(-1, embedding_size)

                        if f"rhs_{ind}" not in candidate_cache:

                            # print("STTEEE MTA")
                            z_scores, rhs_3d = self.get_best_candidates(rel, lhs, None, candidates, last_step)

                            # [Num_queries * Candidates^K]
                            z_scores_1d = z_scores.view(-1)
                            if 'disj' in graph_type or scores_normalize:
                                z_scores_1d = torch.sigmoid(z_scores_1d)

                            # B * S
                            nb_sources = rhs_3d.shape[0] * rhs_3d.shape[1]
                            nb_branches = nb_sources // batch_size
                            if not last_step:
                                batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, candidates).view(-1), t_norm)
                            else:
                                nb_ent = rhs_3d.shape[1]
                                batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, nb_ent).view(-1), t_norm)

                            candidate_cache[f"rhs_{ind}"] = (batch_scores, rhs_3d)

                            if not last_hop:
                                candidate_cache[f"lhs_{indices[hop_num + 1]}"] = (batch_scores, rhs_3d)

                        else:
                            batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
                            candidate_cache[f"lhs_{ind + 1}"] = (batch_scores, rhs_3d)
                            last_hop = True
                            del lhs, rel
                            # #torch.cuda.empty_cache()
                            continue

                        last_hop = True
                        del lhs, rel, rhs, rhs_3d, z_scores_1d, z_scores
                    # #torch.cuda.empty_cache()

                elif 'inter' in inst:
                    ind_1 = int(inst.split("_")[-2])
                    ind_2 = int(inst.split("_")[-1])

                    indices = [ind_1, ind_2]

                    if objective == self.t_norm and dnf_flag:
                        objective = self.t_conorm

                    if len(inst.split("_")) > 3:
                        ind_1 = int(inst.split("_")[-3])
                        ind_2 = int(inst.split("_")[-2])
                        ind_3 = int(inst.split("_")[-1])

                        indices = [ind_1, ind_2, ind_3]

                    for intersection_num, ind in enumerate(indices):
                        last_step = (inst_ind == len(chain_instructions) - 1)  # and ind == indices[0]

                        lhs, rel, rhs = chains[ind]

                        if lhs is not None:
                            lhs = lhs[batch[0]:batch[1]]
                            lhs = lhs.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                            lhs = lhs.view(-1, embedding_size)

                        else:
                            batch_scores, lhs_3d = candidate_cache[f"lhs_{ind}"]
                            lhs = lhs_3d.view(-1, embedding_size)
                            nb_sources = lhs_3d.shape[0] * lhs_3d.shape[1]
                            nb_branches = nb_sources // batch_size

                        rel = rel[batch[0]:batch[1]]
                        rel = rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                        rel = rel.view(-1, embedding_size)

                        if intersection_num > 0 and 'disj' in graph_type:
                            batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
                            rhs = rhs_3d.view(-1, embedding_size)
                            z_scores = self.score_fixed(rel, lhs, rhs, candidates)

                            z_scores_1d = z_scores.view(-1)
                            if 'disj' in graph_type or scores_normalize:
                                z_scores_1d = torch.sigmoid(z_scores_1d)

                            batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores,
                                                                                              t_norm)

                            continue

                        if f"rhs_{ind}" not in candidate_cache or last_step:
                            z_scores, rhs_3d = self.get_best_candidates(rel, lhs, None, candidates, last_step)

                            # [B * Candidates^K] or [B, S-1, N]
                            z_scores_1d = z_scores.view(-1)
                            if 'disj' in graph_type or scores_normalize:
                                z_scores_1d = torch.sigmoid(z_scores_1d)

                            if not last_step:
                                nb_sources = rhs_3d.shape[0] * rhs_3d.shape[1]
                                nb_branches = nb_sources // batch_size

                            if not last_step:
                                batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, candidates).view(-1), t_norm)
                            else:
                                if ind == indices[0]:
                                    nb_ent = rhs_3d.shape[1]
                                else:
                                    nb_ent = 1

                                batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, nb_ent).view(-1), t_norm)
                                nb_ent = rhs_3d.shape[1]

                            candidate_cache[f"rhs_{ind}"] = (batch_scores, rhs_3d)

                            if ind == indices[0] and 'disj' in graph_type:
                                count = len(indices) - 1
                                iterator = 1
                                while count > 0:
                                    candidate_cache[f"rhs_{indices[intersection_num + iterator]}"] = (
                                    batch_scores, rhs_3d)
                                    iterator += 1
                                    count -= 1

                            if ind == indices[-1]:
                                candidate_cache[f"lhs_{ind + 1}"] = (batch_scores, rhs_3d)
                        else:
                            batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
                            candidate_cache[f"rhs_{ind + 1}"] = (batch_scores, rhs_3d)

                            last_hop = True
                            del lhs, rel
                            continue

                        del lhs, rel, rhs, rhs_3d, z_scores_1d, z_scores

        if batch_scores is not None:
            # [B * entites * S ]
            # S ==  K**(V-1)
            scores_2d = batch_scores.view(batch_size, -1, nb_ent)
            res, _ = torch.max(scores_2d, dim=1)
            scores = res if scores is None else torch.cat([scores, res])

            del batch_scores, scores_2d, res, candidate_cache

        else:
            assert False, "Batch Scores are empty: an error went uncaught."

        res = scores

    return res
