import argparse
import ast
import datetime
import math
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import dataloader
import metrics
import mini_batch_test
import utils
from base_model import BaseModel
from evaluate import evaluate_model, log_results


def main_args():
    args = argparse.ArgumentParser(
        description="Bi-DSFA with same-group negative sampling and group-balanced BPR"
    )

    # dataset
    args.add_argument("--dataset_name", default="amazon-book", type=str)
    args.add_argument("--dataset_path", default="OOD_Data", type=str)
    args.add_argument("--result_path", default="OOD_result", type=str)
    args.add_argument("--bpr_num_neg", default=1, type=int)

    # base model
    args.add_argument("--model", default="BiSNA_PGR", type=str)
    args.add_argument("--decay", default=0.0001, type=float)
    args.add_argument("--lr", default=0.001, type=float)
    args.add_argument("--batch_size", default=2048, type=int)
    args.add_argument("--layers_list", default="[2]", type=str)
    args.add_argument("--eps", default=0.2, type=float)
    args.add_argument("--cl_rate_list", default="[20]", type=str)
    args.add_argument("--temperature_list", default="[0.2]", type=str)
    args.add_argument("--seed", default=12345, type=int)

    # alignment
    args.add_argument("--align_reg_list", default="[0.1,50]", type=str)
    args.add_argument("--align_temperature", default=0.2, type=float)

    # train
    args.add_argument("--device", default=3, type=str)
    args.add_argument("--EarlyStop", default=10, type=int)
    args.add_argument("--emb_size", default=64, type=int)
    args.add_argument("--num_epoch", default=400, type=int)
    args.add_argument("--topks", default="[20]", type=str)

    args.add_argument("--warmup_epoch_list", default="[5]", type=str)
    args.add_argument("--knn_k_list", default="[10]", type=str)
    args.add_argument("--sna_sample_size_list", default="[2048]", type=str)

    # group-balanced BPR

    args.add_argument("--num_pop_groups", default=3, type=int)
    args.add_argument("--balanced_neg_per_group", default=1, type=int)
    args.add_argument("--same_group_weight", default="[1.0]", type=str)
    args.add_argument("--balanced_group_weight", default="[1.0]", type=str)
    return args.parse_args()


class BiDSFAGroupBalanced(BaseModel):
    def __init__(self, config, data):
        super(BiDSFAGroupBalanced, self).__init__(config, data)

        self.eps = config.eps
        self.cl_rate = config.cl_rate
        self.temperature = config.temperature
        self.align_temperature = config.align_temperature
        self.warmup_epoch = config.warmup_epoch

        self.item_knn_cache = None
        self.last_item_knn_update = -1
        self.user_knn_cache = None
        self.last_user_knn_update = -1
        self.knn_k = config.knn_k
        self.cache_refresh_interval = 5
        self.sna_sample_size = config.sna_sample_size

        self.num_pop_groups = int(config.num_pop_groups)
        self.balanced_neg_per_group = int(config.balanced_neg_per_group)
        self.num_cross_groups = max(1, self.num_pop_groups - 1)
        self.same_group_weight = float(config.same_group_weight)
        self.balanced_group_weight = float(config.balanced_group_weight)

        pop_list = data.pop_train_count if isinstance(
            data.pop_train_count, list) else data.pop_train_count.tolist()
        self.pop_tensor = torch.tensor(pop_list, device=self.device, dtype=torch.float)

        user_degs = data.active_train_count if isinstance(
            data.active_train_count, list) else data.active_train_count.tolist()
        self.user_degree = torch.tensor(user_degs, device=self.device, dtype=torch.float)

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat(
            [self.user_embeddings.weight, self.item_embeddings.weight], dim=0
        )
        all_emb = []

        for _ in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings)
                ego_embeddings = (
                        ego_embeddings
                        + torch.sign(ego_embeddings)
                        * F.normalize(random_noise, dim=1)
                        * self.eps
                )
            all_emb.append(ego_embeddings)

        if not all_emb:
            all_emb = [ego_embeddings]

        all_emb = torch.stack(all_emb, dim=1)
        all_emb = torch.mean(all_emb, dim=1)
        user_emb, item_emb = torch.split(all_emb, [self.num_users, self.num_items])
        return user_emb, item_emb

    def cl_loss(self, u_idx, i_idx, j_idx):
        u_idx = torch.as_tensor(u_idx, device=self.device).long()
        i_idx = torch.as_tensor(i_idx, device=self.device).long()

        batch_users = torch.unique(u_idx).long()
        batch_items = torch.unique(i_idx).long()

        user_view_1, item_view_1 = self.forward(perturbed=True)
        user_view_2, item_view_2 = self.forward(perturbed=True)

        user_cl_loss = metrics.InfoNCE(
            user_view_1[batch_users],
            user_view_2[batch_users],
            self.temperature,
        ) * self.cl_rate

        item_cl_loss = metrics.InfoNCE(
            item_view_1[batch_items],
            item_view_2[batch_items],
            self.temperature,
        ) * self.cl_rate

        cl_loss = user_cl_loss + item_cl_loss
        return cl_loss, user_cl_loss, item_cl_loss

    def _update_knn_cache(self, emb, cache_name, last_update_name, current_epoch):
        cache = getattr(self, cache_name)
        if cache is not None and (current_epoch - getattr(self, last_update_name)) < self.cache_refresh_interval:
            return

        emb = emb.detach()
        num = emb.shape[0]
        block_size = 2048
        k = self.knn_k
        all_topk_indices = []

        with torch.no_grad():
            query_norm = F.normalize(emb, dim=1)
            target_norm_t = query_norm.t().contiguous()

            for i in range(0, num, block_size):
                end = min(i + block_size, num)
                query_chunk = query_norm[i:end]
                sim = torch.matmul(query_chunk, target_norm_t)
                _, topk_indices = torch.topk(sim, k=min(k + 1, num), dim=1)
                all_topk_indices.append(topk_indices[:, 1:])
                del sim

            filtered = torch.cat(all_topk_indices, dim=0)
            setattr(self, cache_name, filtered)
            setattr(self, last_update_name, current_epoch)

    def batched_contrastive_align_loss_item(self, batch_item_indices, item_emb):
        if self.item_knn_cache is None:
            return torch.tensor(0.0, device=self.device)

        i_idx = torch.unique(torch.as_tensor(batch_item_indices, device=self.device))
        q = F.normalize(item_emb[i_idx], dim=-1)
        knn_idx = self.item_knn_cache[i_idx]
        k_semantic = F.normalize(item_emb[knn_idx], dim=-1).detach()

        with torch.no_grad():
            q_stable = q.detach()
            anchor_sim = torch.sum(q_stable.unsqueeze(1) * k_semantic, dim=-1)
            anchor_weights = F.softmax(anchor_sim / self.align_temperature, dim=1)

        pos_sim = torch.sum(q.unsqueeze(1) * k_semantic, dim=-1)
        weighted_pos_sim = torch.sum(anchor_weights * pos_sim, dim=1)
        row_loss = 1.0 - weighted_pos_sim

        item_pops = self.pop_tensor[i_idx]
        adaptive_weights = 1.0 / torch.log(item_pops + 2.0)
        final_loss = (row_loss * adaptive_weights).mean()
        return final_loss

    def batched_contrastive_align_loss_user(self, batch_user_indices, user_emb):
        if self.user_knn_cache is None:
            return torch.tensor(0.0, device=self.device)

        u_idx = torch.unique(torch.as_tensor(batch_user_indices, device=self.device))
        q = F.normalize(user_emb[u_idx], dim=-1)
        knn_idx = self.user_knn_cache[u_idx]
        k_semantic = F.normalize(user_emb[knn_idx], dim=-1).detach()

        with torch.no_grad():
            q_stable = q.detach()
            anchor_sim = torch.sum(q_stable.unsqueeze(1) * k_semantic, dim=-1)
            anchor_weights = F.softmax(anchor_sim / self.align_temperature, dim=1)

        pos_sim = torch.sum(q.unsqueeze(1) * k_semantic, dim=-1)
        weighted_pos_sim = torch.sum(anchor_weights * pos_sim, dim=1)
        row_loss = 1.0 - weighted_pos_sim

        user_degs = self.user_degree[u_idx]
        adaptive_weights = 1.0 / torch.log(user_degs + 2.0)
        final_loss = (row_loss * adaptive_weights).mean()
        return final_loss

    def group_balanced_bpr_loss(
            self,
            user_emb,
            pos_emb,
            same_neg_emb,
            balanced_neg_emb,
            pos_groups,
            u_idx,
            i_idx,
            same_neg_idx,
            balanced_neg_idx,
    ):
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        same_neg_scores = torch.sum(user_emb * same_neg_emb, dim=1)
        same_group_loss = -F.logsigmoid(pos_scores - same_neg_scores)

        balanced_neg_scores = torch.sum(
            user_emb.unsqueeze(1) * balanced_neg_emb,
            dim=-1,
        )
        balanced_neg_scores = balanced_neg_scores.view(
            -1, self.num_cross_groups, self.balanced_neg_per_group
        )
        balanced_loss = -F.logsigmoid(
            pos_scores.view(-1, 1, 1) - balanced_neg_scores
        )
        balanced_loss = balanced_loss.mean(dim=2).mean(dim=1)

        sample_loss = (
                self.same_group_weight * same_group_loss
                + self.balanced_group_weight * balanced_loss
        )

        group_losses = []
        for group_id in range(self.num_pop_groups):
            group_mask = pos_groups == group_id
            if torch.any(group_mask):
                group_losses.append(sample_loss[group_mask].mean())

        if group_losses:
            bpr_loss = torch.stack(group_losses).mean()
        else:
            bpr_loss = sample_loss.mean()

        user_emb0 = self.user_embeddings.weight[u_idx]
        pos_emb0 = self.item_embeddings.weight[i_idx]
        same_neg_emb0 = self.item_embeddings.weight[same_neg_idx]
        balanced_neg_emb0 = self.item_embeddings.weight[balanced_neg_idx]

        balanced_neg_reg = balanced_neg_emb0.pow(2).sum(dim=2).mean(dim=1)
        reg_term = (
                user_emb0.pow(2).sum(dim=1)
                + pos_emb0.pow(2).sum(dim=1)
                + same_neg_emb0.pow(2).sum(dim=1)
                + balanced_neg_reg
        )
        l2_loss = self.decay * 0.5 * reg_term.mean()
        return bpr_loss, l2_loss

    def batch_loss_group_balanced(
            self,
            u_idx,
            i_idx,
            same_neg_idx,
            balanced_neg_idx,
            pos_groups,
            current_epoch,
            align_reg,
    ):
        u_idx = torch.as_tensor(u_idx, device=self.device).long()
        i_idx = torch.as_tensor(i_idx, device=self.device).long()
        same_neg_idx = torch.as_tensor(same_neg_idx, device=self.device).long()
        balanced_neg_idx = torch.as_tensor(balanced_neg_idx, device=self.device).long()
        pos_groups = torch.as_tensor(pos_groups, device=self.device).long()

        user_embedding, item_embedding = self.forward(perturbed=False)
        user_emb = user_embedding[u_idx]
        pos_emb = item_embedding[i_idx]
        same_neg_emb = item_embedding[same_neg_idx]
        balanced_neg_emb = item_embedding[balanced_neg_idx]

        bpr_loss, l2_loss = self.group_balanced_bpr_loss(
            user_emb,
            pos_emb,
            same_neg_emb,
            balanced_neg_emb,
            pos_groups,
            u_idx,
            i_idx,
            same_neg_idx,
            balanced_neg_idx,
        )

        cl_loss, _, _ = self.cl_loss(u_idx, i_idx, same_neg_idx)

        self._update_knn_cache(
            item_embedding,
            "item_knn_cache",
            "last_item_knn_update",
            current_epoch,
        )
        self._update_knn_cache(
            user_embedding,
            "user_knn_cache",
            "last_user_knn_update",
            current_epoch,
        )

        align_loss = torch.tensor(0.0, device=self.device)
        if current_epoch >= self.warmup_epoch:
            batch_users_unique = torch.unique(u_idx)
            batch_items_unique = torch.unique(i_idx)

            u_sample_len = min(self.sna_sample_size, batch_users_unique.size(0))
            i_sample_len = min(self.sna_sample_size, batch_items_unique.size(0))

            sampled_users = batch_users_unique[
                torch.randperm(batch_users_unique.size(0), device=self.device)[:u_sample_len]
            ]
            sampled_items = batch_items_unique[
                torch.randperm(batch_items_unique.size(0), device=self.device)[:i_sample_len]
            ]

            item_align = (
                    self.batched_contrastive_align_loss_item(sampled_items, item_embedding)
                    * align_reg
            )
            user_align = (
                    self.batched_contrastive_align_loss_user(sampled_users, user_embedding)
                    * align_reg
            )
            align_loss = item_align + user_align

        batch_loss = bpr_loss + l2_loss + cl_loss + align_loss
        return batch_loss, bpr_loss, l2_loss, cl_loss, align_loss


def test(model):
    user_embedding, item_embedding = model.forward()
    return user_embedding.detach().cpu().numpy(), item_embedding.detach().cpu().numpy()


def train(config, data, model, optimizer, early_stopping, logger, sampler):
    model.train()
    cuda_device = torch.device(model.device) if not isinstance(model.device, torch.device) else model.device
    track_gpu_memory = torch.cuda.is_available() and cuda_device.type == "cuda"

    for epoch in range(config.num_epoch):
        if track_gpu_memory:
            torch.cuda.reset_peak_memory_stats(cuda_device)
            torch.cuda.synchronize(cuda_device)

        start = datetime.datetime.now()
        train_res = {
            "bpr_loss": 0.0,
            "emb_loss": 0.0,
            "batch_loss": 0.0,
            "cl_loss": 0.0,
            "align_loss": 0.0,
        }

        current_align_reg = config.align_reg if epoch >= config.warmup_epoch else 0.0
        total_batches = math.ceil(len(data.training_data) / config.batch_size)

        with tqdm(total=total_batches, desc=f"Epoch {epoch}", unit="batch") as pbar:
            for batch in dataloader.next_batch_group_balanced(data, config.batch_size, sampler):
                user_idx, pos_idx, same_neg_idx, balanced_neg_idx, pos_groups = batch

                batch_loss, bpr_loss, l2_loss, cl_loss, align_loss = (
                    model.batch_loss_group_balanced(
                        user_idx,
                        pos_idx,
                        same_neg_idx,
                        balanced_neg_idx,
                        pos_groups,
                        epoch,
                        current_align_reg,
                    )
                )

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                train_res["bpr_loss"] += bpr_loss.item()
                train_res["emb_loss"] += l2_loss.item()
                train_res["batch_loss"] += batch_loss.item()
                train_res["cl_loss"] += cl_loss.item()
                train_res["align_loss"] += align_loss.item()

                pbar.set_postfix({"loss": batch_loss.item()})
                pbar.update(1)

        for key in train_res:
            train_res[key] /= total_batches

        training_logs = f"epoch: {epoch}, "
        for name, value in train_res.items():
            training_logs += f"{name}:{value:.6f} "
        logger.info(training_logs)

        if track_gpu_memory:
            torch.cuda.synchronize(cuda_device)
            peak_gpu_mem_mb = torch.cuda.max_memory_allocated(cuda_device) / 1024 / 1024
        else:
            peak_gpu_mem_mb = 0.0

        train_time = datetime.datetime.now()
        train_time_sec = (train_time - start).total_seconds()

        model.eval()
        user_embedding, item_embedding = test(model)
        val_hr, val_recall, val_ndcg = mini_batch_test.test_acc_batch(
            data.val_U2I,
            data.train_U2I,
            user_embedding,
            item_embedding,
        )
        logger.info(
            "val_hr@20:{:.6f}   val_recall@20:{:.6f}   val_ndcg@20:{:.6f}"
            "   train_time:{:.2f}s   test_time:{:.2f}s   peak_gpu_mem:{:.2f}MB".format(
                val_hr,
                val_recall,
                val_ndcg,
                train_time_sec,
                (datetime.datetime.now() - train_time).total_seconds(),
                peak_gpu_mem_mb,
            )
        )
        logger.info(
            "efficiency epoch:{} train_time_sec:{:.2f} peak_gpu_mem_mb:{:.2f} peak_gpu_mem_gb:{:.4f}".format(
                epoch,
                train_time_sec,
                peak_gpu_mem_mb,
                peak_gpu_mem_mb / 1024.0,
            )
        )

        early_stopping(val_ndcg, model, epoch)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break


def log_bucket_stats(logger, sampler):
    for stat in sampler.group_stats:
        logger.info(
            "group_id:%d size:%d min_pop:%.2f mean_pop:%.2f max_pop:%.2f"
            % (
                stat["group_id"],
                stat["size"],
                stat["min_pop"],
                stat["mean_pop"],
                stat["max_pop"],
            )
        )


def main(config):
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M%S")
    file_name = "_".join(
        (
            str(config.align_reg),
            str(config.layers),
            str(config.cl_rate),
            str(config.knn_k),
            f"groups{config.num_pop_groups}",
            f"neg{config.balanced_neg_per_group}",
            f"sgw{config.same_group_weight}",
            f"bgw{config.balanced_group_weight}",
            timestamp,
        )
    )

    result_path = os.path.join(config.result_path, config.model, config.dataset_name, file_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    logger_file_name = os.path.join(result_path, "train_logger")
    logger = utils.get_logger(logger_file_name)
    for name, value in vars(config).items():
        logger.info("%20s =======> %-20s" % (name, value))

    if config.seed:
        utils.setup_seed(config.seed)

    data = dataloader.Data(config, logger)
    data.norm_adj = dataloader.LaplaceGraph(
        data.num_users,
        data.num_items,
        data.train_U2I,
    ).generate()

    sampler = dataloader.PopularityBucketSampler(
        data,
        config.num_pop_groups,
        config.balanced_neg_per_group,
        config.seed,
    )
    log_bucket_stats(logger, sampler)

    model = BiDSFAGroupBalanced(config, data)
    model.to(model.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    early_stopping = utils.EarlyStopping(
        logger,
        config.EarlyStop,
        verbose=True,
        path=result_path,
    )

    train(config, data, model, optimizer, early_stopping, logger, sampler)

    model.load_state_dict(torch.load(result_path + "/best_val_epoch.pt", map_location=model.device))


    model.eval()
    user_embedding, item_embedding = test(model)


    val_hr, val_recall, val_ndcg = mini_batch_test.test_acc_batch(
        data.val_U2I, data.train_U2I, user_embedding, item_embedding, topk=20)

    test_OOD_hr, test_OOD_recall, test_OOD_ndcg = mini_batch_test.test_acc_batch(
        data.test_U2I, data.train_U2I, user_embedding, item_embedding, topk=20)


    if hasattr(data, 'test_iid_U2I'):
        test_IID_hr, test_IID_recall, test_IID_ndcg = mini_batch_test.test_acc_batch(
            data.test_iid_U2I, data.train_U2I, user_embedding, item_embedding, topk=20)
    else:
        test_IID_hr, test_IID_recall, test_IID_ndcg = test_OOD_hr, test_OOD_recall, test_OOD_ndcg


    log_results(logger, 20, val_hr, val_recall, val_ndcg, test_OOD_hr, test_OOD_recall, test_OOD_ndcg, test_IID_hr,
                test_IID_recall, test_IID_ndcg)

    return (
        val_hr,
        val_recall,
        val_ndcg,
        test_OOD_hr, test_OOD_recall, test_OOD_ndcg, test_IID_hr, test_IID_recall, test_IID_ndcg,
        result_path,
    )


if __name__ == "__main__":
    config = main_args()
    config.device = utils.normalize_device(config.device)
    same_group_weight_candidates = utils.parse_search_values(config.same_group_weight)
    balanced_group_weight_candidates = utils.parse_search_values(config.balanced_group_weight)

    result_path = os.path.join(config.result_path, config.model, config.dataset_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    summary_path = os.path.join(result_path, "best_performance_group_balanced.txt")
    with open(summary_path, "a+", encoding="utf-8") as f:
        for cl_rate in ast.literal_eval(config.cl_rate_list):
            for layers in ast.literal_eval(config.layers_list):
                for align_reg in ast.literal_eval(config.align_reg_list):
                    for temperature in ast.literal_eval(config.temperature_list):
                        for knn_k in ast.literal_eval(config.knn_k_list):
                            for warmup_epoch in ast.literal_eval(config.warmup_epoch_list):
                                for sna_sample_size in ast.literal_eval(config.sna_sample_size_list):
                                    for same_group_weight in same_group_weight_candidates:
                                        for balanced_group_weight in balanced_group_weight_candidates:
                                            config.sna_sample_size = sna_sample_size
                                            config.warmup_epoch = warmup_epoch
                                            config.knn_k = knn_k
                                            config.temperature = temperature
                                            config.cl_rate = cl_rate
                                            config.layers = layers
                                            config.align_reg = align_reg
                                            config.same_group_weight = float(same_group_weight)
                                            config.balanced_group_weight = float(balanced_group_weight)

                                            (
                                                val_hr,
                                                val_recall,
                                                val_ndcg,
                                                test_ood_hr,
                                                test_ood_recall,
                                                test_ood_ndcg,
                                                test_iid_hr,
                                                test_iid_recall,
                                                test_iid_ndcg,
                                                run_result_path,
                                            ) = main(config)

                                            f.write("\n")
                                            f.write(
                                                f"\n ====layers:{config.layers}, cl_rate:{config.cl_rate},"
                                                f" K:{config.knn_k}, groups:{config.num_pop_groups},"
                                                f" neg_per_group:{config.balanced_neg_per_group}\n"
                                                f" same_group_weight:{config.same_group_weight},"
                                                f" balanced_group_weight:{config.balanced_group_weight}\n"
                                                f" align_reg:{config.align_reg}\n"
                                                f" best_hr@20:{val_hr:.6f}=====best_recall@20:{val_recall:.6f}"
                                                f"====best_ndcg@20:{val_ndcg:.6f}\n"
                                                f" test_OOD_hr@20:{test_ood_hr:.6f}"
                                                f"   test_OOD_recall@20:{test_ood_recall:.6f}"
                                                f"   test_OOD_ndcg@20:{test_ood_ndcg:.6f}\n"
                                                f" test_IID_hr@20:{test_iid_hr:.6f}"
                                                f"   test_IID_recall@20:{test_iid_recall:.6f}"
                                                f"   test_IID_ndcg@20:{test_iid_ndcg:.6f}\n"
                                                f" Result_path:{run_result_path}\n"
                                            )
                                            f.write("\n")
