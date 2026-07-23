import math

import faiss
import numba as nb
import numpy as np
import torch


def _to_numpy_float32(array_like):
    if isinstance(array_like, torch.Tensor):
        array_like = array_like.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(array_like, dtype=np.float32))



def _normalize_group_percentiles(group_percentiles):
    if group_percentiles is None:
        return None

    percentiles = [float(value) for value in group_percentiles]
    if not percentiles:
        raise ValueError("group_percentiles 不能为空。")

    if max(percentiles) > 1.0:
        percentiles = [value / 100.0 for value in percentiles]

    previous = 0.0
    for value in percentiles:
        if value <= 0.0 or value > 1.0:
            raise ValueError("group_percentiles 必须位于 (0, 1] 或 (0, 100]。")
        if value <= previous:
            raise ValueError("group_percentiles 必须严格递增。")
        previous = value

    if abs(percentiles[-1] - 1.0) > 1e-8:
        raise ValueError("group_percentiles 最后一个边界必须是 1.0 或 100。")

    return percentiles


def build_item_popularity_groups(pop_train_count, num_groups=3, group_percentiles=None):
    pop_array = np.asarray(pop_train_count, dtype=np.float32)
    sorted_items = np.argsort(pop_array, kind="mergesort")
    normalized_percentiles = _normalize_group_percentiles(group_percentiles)

    if normalized_percentiles is None:
        num_groups = max(2, int(num_groups))
        bucket_splits = np.array_split(sorted_items, num_groups)
        percentile_ranges = [
            (group_id / num_groups, (group_id + 1) / num_groups)
            for group_id in range(num_groups)
        ]
    else:
        num_groups = len(normalized_percentiles)
        bucket_splits = []
        percentile_ranges = []
        total_items = len(sorted_items)
        previous_idx = 0
        previous_pct = 0.0

        for percentile in normalized_percentiles:
            end_idx = int(round(percentile * total_items))
            end_idx = max(previous_idx, min(end_idx, total_items))
            bucket_splits.append(sorted_items[previous_idx:end_idx])
            percentile_ranges.append((previous_pct, percentile))
            previous_idx = end_idx
            previous_pct = percentile

    item_group = np.zeros(len(pop_array), dtype=np.int64)
    group_stats = []

    for group_id, item_ids in enumerate(bucket_splits):
        item_ids = np.asarray(item_ids, dtype=np.int64)
        start_pct, end_pct = percentile_ranges[group_id]
        if item_ids.size > 0:
            item_group[item_ids] = group_id
            group_stats.append(
                {
                    "group_id": int(group_id),
                    "start_percentile": float(start_pct),
                    "end_percentile": float(end_pct),
                    "size": int(item_ids.size),
                    "min_pop": float(pop_array[item_ids].min()),
                    "mean_pop": float(pop_array[item_ids].mean()),
                    "max_pop": float(pop_array[item_ids].max()),
                }
            )
        else:
            group_stats.append(
                {
                    "group_id": int(group_id),
                    "start_percentile": float(start_pct),
                    "end_percentile": float(end_pct),
                    "size": 0,
                    "min_pop": 0.0,
                    "mean_pop": 0.0,
                    "max_pop": 0.0,
                }
            )

    return item_group, group_stats


def default_group_names(num_groups):
    if int(num_groups) == 3:
        return ["tail", "mid", "head"]
    return [f"group_{group_id}" for group_id in range(int(num_groups))]


def _build_filtered_topk(raw_ranked_items, masked_items, topk):
    masked_set = set(int(item_id) for item_id in masked_items)
    filtered_items = []
    for item_id in raw_ranked_items:
        item_id = int(item_id)
        if item_id in masked_set:
            continue
        filtered_items.append(item_id)
        if len(filtered_items) == topk:
            break
    return filtered_items


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


def test_acc_batch_by_item_groups(
        _test_U2I,
        _train_U2I,
        _user_matrix,
        _item_matrix,
        item_group,
        group_stats=None,
        topk=20,
        group_names=None,
):
    test_users = list(_test_U2I.keys())
    if not test_users:
        return []

    item_group = np.asarray(item_group, dtype=np.int64)
    num_groups = int(item_group.max()) + 1 if item_group.size > 0 else 0
    if group_names is None:
        group_names = default_group_names(num_groups)
    else:
        group_names = list(group_names)
        if len(group_names) < num_groups:
            group_names.extend(
                [f"group_{group_id}" for group_id in range(len(group_names), num_groups)]
            )

    if group_stats is None:
        group_stats = [{"group_id": group_id, "size": 0, "min_pop": 0.0, "mean_pop": 0.0, "max_pop": 0.0}
                       for group_id in range(num_groups)]

    group_results = []
    for group_id in range(num_groups):
        stats = group_stats[group_id] if group_id < len(group_stats) else {}
        group_results.append(
            {
                "group_id": int(group_id),
                "group_name": group_names[group_id],
                "start_percentile": float(stats.get("start_percentile", 0.0)),
                "end_percentile": float(stats.get("end_percentile", 0.0)),
                "catalog_size": int(stats.get("size", 0)),
                "min_pop": float(stats.get("min_pop", 0.0)),
                "mean_pop": float(stats.get("mean_pop", 0.0)),
                "max_pop": float(stats.get("max_pop", 0.0)),
                "user_count": 0,
                "positive_count": 0,
                "hit_count": 0,
                "hr_sum": 0.0,
                "recall_sum": 0.0,
                "ndcg_sum": 0.0,
            }
        )

    user_matrix = _to_numpy_float32(_user_matrix)
    item_matrix = _to_numpy_float32(_item_matrix)
    index = faiss.IndexFlatIP(item_matrix.shape[-1])
    index.add(item_matrix)

    batch_size = 35000
    for start in range(0, len(test_users), batch_size):
        batch_users = test_users[start:start + batch_size]
        batch_test = {user_id: _test_U2I[user_id] for user_id in batch_users}
        batch_train = {user_id: _train_U2I[user_id] for user_id in batch_users}

        traindata = [np.asarray(batch_train[user_id], dtype=np.int64) for user_id in batch_users]
        max_mask_items_length = max((len(items) for items in traindata), default=0)
        search_k = min(topk + max_mask_items_length, item_matrix.shape[0])
        query_vectors = np.ascontiguousarray(user_matrix[batch_users])
        _, user_rank_pred_items = index.search(query_vectors, search_k)

        for row_idx, user_id in enumerate(batch_users):
            filtered_topk = _build_filtered_topk(
                user_rank_pred_items[row_idx],
                batch_train[user_id],
                topk,
            )
            if not filtered_topk:
                continue

            positives_by_group = {}
            for item_id in batch_test[user_id]:
                group_id = int(item_group[int(item_id)])
                positives_by_group.setdefault(group_id, []).append(int(item_id))

            for group_id, positives in positives_by_group.items():
                pos_length = len(positives)
                if pos_length == 0:
                    continue

                test_item_set = set(positives)
                hit_value = 0
                dcg_value = 0.0

                for rank_idx, item_id in enumerate(filtered_topk):
                    if item_id in test_item_set:
                        hit_value += 1
                        dcg_value += math.log(2.0) / math.log(rank_idx + 2.0)

                target_length = min(topk, pos_length)
                idcg = 0.0
                for rank_idx in range(target_length):
                    idcg += math.log(2.0) / math.log(rank_idx + 2.0)

                group_results[group_id]["user_count"] += 1
                group_results[group_id]["positive_count"] += pos_length
                group_results[group_id]["hit_count"] += hit_value
                group_results[group_id]["hr_sum"] += hit_value / target_length
                group_results[group_id]["recall_sum"] += hit_value / pos_length
                group_results[group_id]["ndcg_sum"] += dcg_value / idcg if idcg > 0 else 0.0

    for result in group_results:
        user_count = result["user_count"]
        if user_count > 0:
            result["hr"] = result["hr_sum"] / user_count
            result["recall"] = result["recall_sum"] / user_count
            result["ndcg"] = result["ndcg_sum"] / user_count
        else:
            result["hr"] = 0.0
            result["recall"] = 0.0
            result["ndcg"] = 0.0

        positive_count = result["positive_count"]
        result["avg_pos_per_user"] = (
            positive_count / user_count if user_count > 0 else 0.0
        )
        result["micro_recall"] = (
            result["hit_count"] / positive_count if positive_count > 0 else 0.0
        )

        del result["hr_sum"]
        del result["recall_sum"]
        del result["ndcg_sum"]

    return group_results


def recommendation_popularity_distribution(
        _eval_U2I,
        _train_U2I,
        _user_matrix,
        _item_matrix,
        pop_train_count,
        item_group,
        group_stats=None,
        topk=20,
        group_names=None,
):
    eval_users = list(_eval_U2I.keys())
    if not eval_users:
        return {
            "arp": 0.0,
            "avg_log_pop": 0.0,
            "user_count": 0,
            "recommendation_count": 0,
            "group_distribution": [],
        }

    pop_array = np.asarray(pop_train_count, dtype=np.float32)
    item_group = np.asarray(item_group, dtype=np.int64)
    num_groups = int(item_group.max()) + 1 if item_group.size > 0 else 0

    if group_names is None:
        group_names = default_group_names(num_groups)
    else:
        group_names = list(group_names)
        if len(group_names) < num_groups:
            group_names.extend(
                [f"group_{group_id}" for group_id in range(len(group_names), num_groups)]
            )

    if group_stats is None:
        group_stats = [{"group_id": group_id, "size": 0, "min_pop": 0.0, "mean_pop": 0.0, "max_pop": 0.0}
                       for group_id in range(num_groups)]

    group_distribution = []
    for group_id in range(num_groups):
        stats = group_stats[group_id] if group_id < len(group_stats) else {}
        group_distribution.append(
            {
                "group_id": int(group_id),
                "group_name": group_names[group_id],
                "start_percentile": float(stats.get("start_percentile", 0.0)),
                "end_percentile": float(stats.get("end_percentile", 0.0)),
                "catalog_size": int(stats.get("size", 0)),
                "min_pop": float(stats.get("min_pop", 0.0)),
                "mean_pop": float(stats.get("mean_pop", 0.0)),
                "max_pop": float(stats.get("max_pop", 0.0)),
                "recommend_count": 0,
                "recommend_ratio": 0.0,
                "avg_recommend_pop": 0.0,
            }
        )

    user_matrix = _to_numpy_float32(_user_matrix)
    item_matrix = _to_numpy_float32(_item_matrix)
    index = faiss.IndexFlatIP(item_matrix.shape[-1])
    index.add(item_matrix)

    total_recommend_count = 0
    total_pop_sum = 0.0
    total_log_pop_sum = 0.0
    batch_size = 35000

    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start:start + batch_size]
        batch_train = {user_id: _train_U2I[user_id] for user_id in batch_users}

        traindata = [np.asarray(batch_train[user_id], dtype=np.int64) for user_id in batch_users]
        max_mask_items_length = max((len(items) for items in traindata), default=0)
        search_k = min(topk + max_mask_items_length, item_matrix.shape[0])
        query_vectors = np.ascontiguousarray(user_matrix[batch_users])
        _, user_rank_pred_items = index.search(query_vectors, search_k)

        for row_idx, user_id in enumerate(batch_users):
            filtered_topk = _build_filtered_topk(
                user_rank_pred_items[row_idx],
                batch_train[user_id],
                topk,
            )
            for item_id in filtered_topk:
                item_id = int(item_id)
                pop_value = float(pop_array[item_id])
                group_id = int(item_group[item_id])

                total_recommend_count += 1
                total_pop_sum += pop_value
                total_log_pop_sum += math.log(pop_value + 1.0)

                group_distribution[group_id]["recommend_count"] += 1
                group_distribution[group_id]["avg_recommend_pop"] += pop_value

    for result in group_distribution:
        recommend_count = result["recommend_count"]
        if recommend_count > 0:
            result["avg_recommend_pop"] = result["avg_recommend_pop"] / recommend_count
        else:
            result["avg_recommend_pop"] = 0.0
        result["recommend_ratio"] = (
            recommend_count / total_recommend_count if total_recommend_count > 0 else 0.0
        )

    return {
        "arp": total_pop_sum / total_recommend_count if total_recommend_count > 0 else 0.0,
        "avg_log_pop": total_log_pop_sum / total_recommend_count if total_recommend_count > 0 else 0.0,
        "user_count": len(eval_users),
        "recommendation_count": total_recommend_count,
        "group_distribution": group_distribution,
    }
