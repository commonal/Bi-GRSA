import math

import faiss
import numba as nb
import numpy as np
import torch


def _to_numpy_float32(array_like):
    if isinstance(array_like, torch.Tensor):
        array_like = array_like.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(array_like, dtype=np.float32))


@nb.njit(nopython=True)
def compute_ranking_metrics(testdata, traindata, user_rank_pred_items, topk=20):
    hr_sum = 0.0
    recall_sum = 0.0
    ndcg_sum = 0.0

    for i in range(len(testdata)):
        test_items = testdata[i]
        pos_length = len(test_items)
        if pos_length == 0:
            continue

        mask_items = traindata[i]
        max_candidate_length = min(len(mask_items) + topk, user_rank_pred_items.shape[1])
        hit_value = 0
        dcg_value = 0.0
        pred_count = 0

        for candidate_idx in range(max_candidate_length):
            item = user_rank_pred_items[i, candidate_idx]

            masked = False
            for mask_idx in range(len(mask_items)):
                if item == mask_items[mask_idx]:
                    masked = True
                    break
            if masked:
                continue

            for test_idx in range(pos_length):
                if item == test_items[test_idx]:
                    hit_value += 1
                    dcg_value += math.log(2.0) / math.log(pred_count + 2.0)
                    break

            pred_count += 1
            if pred_count == topk:
                break

        target_length = min(topk, pos_length)
        idcg = 0.0
        for rank_idx in range(target_length):
            idcg += math.log(2.0) / math.log(rank_idx + 2.0)

        hr_sum += hit_value / target_length
        recall_sum += hit_value / pos_length
        ndcg_sum += dcg_value / idcg

    return hr_sum, recall_sum, ndcg_sum


def num_faiss_evaluate(_test_ratings, _train_ratings, _user_matrix, _item_matrix, Topk=20, index=None):
    """
    Evaluation for ranking results.
    Keep the same metric definition as before, but avoid rebuilding the
    retrieval index and avoid querying all users for each batch.
    """
    test_users = list(_test_ratings.keys())
    if not test_users:
        return 0.0, 0.0, 0.0

    user_matrix = _to_numpy_float32(_user_matrix)
    item_matrix = _to_numpy_float32(_item_matrix)
    query_vectors = np.ascontiguousarray(user_matrix[test_users])

    if index is None:
        dim = item_matrix.shape[-1]
        index = faiss.IndexFlatIP(dim)
        index.add(item_matrix)

    testdata = nb.typed.List(
        [np.asarray(_test_ratings[user], dtype=np.int64) for user in test_users]
    )
    traindata = nb.typed.List(
        [np.asarray(_train_ratings[user], dtype=np.int64) for user in test_users]
    )

    max_mask_items_length = max(len(items) for items in traindata)
    search_k = min(Topk + max_mask_items_length, item_matrix.shape[0])
    _, user_rank_pred_items = index.search(query_vectors, search_k)

    hr_out, recall_out, ndcg_out = compute_ranking_metrics(
        testdata, traindata, user_rank_pred_items, topk=Topk
    )
    return hr_out, recall_out, ndcg_out


def test_acc_batch(_test_U2I, _train_U2I, _user_matrix, _item_matrix, topk=20):
    test_users = list(_test_U2I.keys())
    data_size = len(test_users)
    if data_size == 0:
        return 0.0, 0.0, 0.0

    batch_id = 0
    batch_size = 35000
    hr_all, recall_all, ndcg_all = 0.0, 0.0, 0.0

    user_matrix = _to_numpy_float32(_user_matrix)
    item_matrix = _to_numpy_float32(_item_matrix)
    index = faiss.IndexFlatIP(item_matrix.shape[-1])
    index.add(item_matrix)

    while batch_id < data_size:
        batch_end = min(batch_id + batch_size, data_size)
        batch_users = test_users[batch_id:batch_end]
        batch_id = batch_end

        batch_test = {key: _test_U2I[key] for key in batch_users}
        batch_train = {key: _train_U2I[key] for key in batch_users}
        hr_out, recall_out, ndcg_out = num_faiss_evaluate(
            batch_test,
            batch_train,
            user_matrix,
            item_matrix,
            Topk=topk,
            index=index,
        )
        hr_all += hr_out
        recall_all += recall_out
        ndcg_all += ndcg_out

    hr_all = hr_all / data_size
    recall_all = recall_all / data_size
    ndcg_all = ndcg_all / data_size
    return hr_all, recall_all, ndcg_all
