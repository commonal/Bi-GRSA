import math

import torch


def evaluate_split(model, eval_u2i, train_u2i, topk, batch_size=512):
    model.eval()
    eval_users = list(eval_u2i.keys())
    if not eval_users:
        return 0.0, 0.0, 0.0

    hr_sum, recall_sum, ndcg_sum = 0.0, 0.0, 0.0
    valid_user_count = 0

    with torch.no_grad():
        for start in range(0, len(eval_users), batch_size):
            batch_users = eval_users[start:start + batch_size]
            scores = model.full_sort_scores(batch_users).detach().cpu()

            for row_idx, user_id in enumerate(batch_users):
                history = train_u2i.get(user_id, [])
                if history:
                    scores[row_idx, history] = float("-inf")

            k = min(int(topk), scores.size(1))
            top_items = torch.topk(scores, k=k, dim=1).indices.tolist()

            for row_idx, user_id in enumerate(batch_users):
                test_items = eval_u2i[user_id]
                pos_length = len(test_items)
                if pos_length == 0:
                    continue

                test_item_set = set(test_items)
                hit_value = 0
                dcg_value = 0.0

                for rank_idx, item_id in enumerate(top_items[row_idx]):
                    if item_id in test_item_set:
                        hit_value += 1
                        dcg_value += math.log(2.0) / math.log(rank_idx + 2.0)

                target_length = min(k, pos_length)
                idcg = 0.0
                for rank_idx in range(target_length):
                    idcg += math.log(2.0) / math.log(rank_idx + 2.0)

                hr_sum += hit_value / target_length
                recall_sum += hit_value / pos_length
                ndcg_sum += dcg_value / idcg if idcg > 0 else 0.0
                valid_user_count += 1

    if valid_user_count == 0:
        return 0.0, 0.0, 0.0

    return (
        hr_sum / valid_user_count,
        recall_sum / valid_user_count,
        ndcg_sum / valid_user_count,
    )


def evaluate_model(model, data, topk, logger):
    topk = int(topk)

    val_hr, val_recall, val_ndcg = evaluate_split(
        model, data.val_U2I, data.train_U2I, topk
    )
    test_OOD_hr, test_OOD_recall, test_OOD_ndcg = evaluate_split(
        model, data.test_U2I, data.train_U2I, topk
    )
    test_IID_hr, test_IID_recall, test_IID_ndcg = evaluate_split(
        model, data.test_iid_U2I, data.train_U2I, topk
    )

    logger.info(
        "Validation@{}: hr={:.4f}, recall={:.4f}, ndcg={:.4f}".format(
            topk, val_hr, val_recall, val_ndcg
        )
    )
    logger.info(
        "OOD Test@{}: hr={:.4f}, recall={:.4f}, ndcg={:.4f}".format(
            topk, test_OOD_hr, test_OOD_recall, test_OOD_ndcg
        )
    )
    logger.info(
        "IID Test@{}: hr={:.4f}, recall={:.4f}, ndcg={:.4f}".format(
            topk, test_IID_hr, test_IID_recall, test_IID_ndcg
        )
    )

    return (
        val_hr,
        val_recall,
        val_ndcg,
        test_OOD_hr,
        test_OOD_recall,
        test_OOD_ndcg,
        test_IID_hr,
        test_IID_recall,
        test_IID_ndcg,
    )


def log_results(
    logger,
    topk,
    val_hr,
    val_recall,
    val_ndcg,
    test_OOD_hr,
    test_OOD_recall,
    test_OOD_ndcg,
    test_IID_hr,
    test_IID_recall,
    test_IID_ndcg,
):
    logger.info(
        "=======Best Validation performance=====\n"
        "val_hr@{}:{:.6f}   val_recall@{}:{:.6f}   val_ndcg@{}:{:.6f} ".format(
            topk, val_hr, topk, val_recall, topk, val_ndcg
        )
    )
    logger.info(
        "=======OOD performance=====\n"
        "test_OOD_hr@{}:{:.6f}   test_OOD_recall@{}:{:.6f}   test_OOD_ndcg@{}:{:.6f} ".format(
            topk, test_OOD_hr, topk, test_OOD_recall, topk, test_OOD_ndcg
        )
    )
    logger.info(
        "=======IID performance=====\n"
        "test_IID_hr@{}:{:.6f}   test_IID_recall@{}:{:.6f}   test_IID_ndcg@{}:{:.6f} ".format(
            topk, test_IID_hr, topk, test_IID_recall, topk, test_IID_ndcg
        )
    )
