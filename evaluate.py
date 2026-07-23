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
