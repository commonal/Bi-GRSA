import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

NEW_STYLE_DATASETS = {"amazon-book.new", "tencent.new", "yelp2018.new"}
VAL_ONLY_DATASETS = {"Ciao", "douban", "amazon_art", "yahoo.new"}


def _resolve_split_file(dataset_path, primary_name, fallback_names=None):
    primary_path = os.path.join(dataset_path, primary_name)
    if os.path.exists(primary_path):
        return primary_path

    for fallback_name in fallback_names or []:
        fallback_path = os.path.join(dataset_path, fallback_name)
        if os.path.exists(fallback_path):
            return fallback_path

    checked_names = [primary_name] + list(fallback_names or [])
    raise FileNotFoundError(
        f"None of the expected split files exist in {dataset_path}: {checked_names}"
    )


def _parse_interaction_line(line):
    tokens = str(line).strip().split()
    if not tokens:
        return None, []
    values = [int(token) for token in tokens]
    return int(values[0]), values[1:]


def _read_user_item_split(file_path, collect_pairs=False):
    user_to_items = defaultdict(list)
    item_to_users = defaultdict(list)
    interaction_pairs = []
    user_ids = []
    item_ids = []
    max_user_id = -1
    max_item_id = -1
    interaction_count = 0

    split_frame = pd.read_table(file_path, header=None)
    for raw_line in split_frame.iloc[:, 0]:
        user_id, items = _parse_interaction_line(raw_line)
        if user_id is None:
            continue

        max_user_id = max(max_user_id, user_id)
        if not items:
            continue

        max_item_id = max(max_item_id, max(items))
        interaction_count += len(items)
        user_ids.extend([user_id] * len(items))
        item_ids.extend(items)
        user_to_items[user_id].extend(items)

        for item_id in items:
            item_to_users[item_id].append(user_id)
            if collect_pairs:
                interaction_pairs.append([user_id, item_id])

    return (
        user_to_items,
        item_to_users,
        interaction_pairs,
        user_ids,
        item_ids,
        max_user_id,
        max_item_id,
        interaction_count,
    )


def _build_frequency_list(indices, size):
    if size <= 0:
        return []
    if not indices:
        return [0] * size
    counts = np.bincount(np.asarray(indices, dtype=np.int64), minlength=size)
    return counts.astype(np.int64).tolist()


class Data(object):
    def __init__(self, config, logger):
        self.logger = logger
        self.dataset_name = config.dataset_name
        self.dataset_path = os.path.join(config.dataset_path, config.dataset_name)
        self.num_neg = config.bpr_num_neg

        (
            self.num_users,
            self.num_items,
            self.train_U2I,
            self.training_data,
            self.test_U2I,
            self.active_train_count,
            self.pop_train_count,
            total_interactions,
            self.test_I2U,
            self.train_I2U,
            self.val_U2I,
            self.test_iid_U2I,
            self.val_sum,
            self.test_iid_sum,
        ) = self.load_data()

        logger.info(
            "num_users:{:d}   num_items:{:d}   density:{:.6f}%".format(
                self.num_users,
                self.num_items,
                total_interactions / self.num_items / self.num_users * 100,
            )
        )

    def load_data(self):
        train_path = _resolve_split_file(self.dataset_path, "train.txt")
        test_path = _resolve_split_file(
            self.dataset_path, "test.txt", fallback_names=["test_ood.txt"]
        )

        (
            train_U2I,
            train_I2U,
            training_data,
            train_users,
            train_items,
            train_max_user,
            train_max_item,
            train_num,
        ) = _read_user_item_split(train_path, collect_pairs=True)
        (
            test_U2I,
            test_I2U,
            _,
            _,
            _,
            test_max_user,
            test_max_item,
            test_num,
        ) = _read_user_item_split(test_path, collect_pairs=False)

        num_users = max(train_max_user, test_max_user)
        num_items = max(train_max_item, test_max_item)

        if self.dataset_name in NEW_STYLE_DATASETS:
            val_path = _resolve_split_file(
                self.dataset_path,
                "valid.txt",
                fallback_names=["val.txt", "test.txt", "test_ood.txt"],
            )
            test_iid_path = _resolve_split_file(
                self.dataset_path, "test_id.txt", fallback_names=["test_iid.txt"]
            )

            (
                val_U2I,
                _,
                _,
                _,
                _,
                val_max_user,
                val_max_item,
                val_sum,
            ) = _read_user_item_split(val_path, collect_pairs=False)
            (
                test_iid_U2I,
                _,
                _,
                _,
                _,
                test_iid_max_user,
                test_iid_max_item,
                test_iid_sum,
            ) = _read_user_item_split(test_iid_path, collect_pairs=False)

            num_users = max(num_users, val_max_user, test_iid_max_user)
            num_items = max(num_items, val_max_item, test_iid_max_item)
        elif self.dataset_name in VAL_ONLY_DATASETS:
            val_path = _resolve_split_file(
                self.dataset_path, "val.txt", fallback_names=["valid.txt"]
            )
            (
                val_U2I,
                _,
                _,
                _,
                _,
                val_max_user,
                val_max_item,
                val_sum,
            ) = _read_user_item_split(val_path, collect_pairs=False)
            test_iid_U2I = test_U2I
            test_iid_sum = 0

            num_users = max(num_users, val_max_user)
            num_items = max(num_items, val_max_item)
        else:
            val_U2I = test_U2I
            test_iid_U2I = test_U2I
            val_sum = 0
            test_iid_sum = 0

        num_users += 1
        num_items += 1

        if self.num_neg > 1:
            training_data = [pair for pair in training_data for _ in range(self.num_neg)]

        active_train = _build_frequency_list(train_users, num_users)
        pop_train = _build_frequency_list(train_items, num_items)
        total_interactions = train_num + test_num + val_sum + test_iid_sum

        return (
            num_users,
            num_items,
            train_U2I,
            training_data,
            test_U2I,
            active_train,
            pop_train,
            total_interactions,
            test_I2U,
            train_I2U,
            val_U2I,
            test_iid_U2I,
            val_sum,
            test_iid_sum,
        )

class Graph(object):
    def __init__(self, num_users, num_items, train_U2I, gama):
        self.num_users = num_users
        self.num_items = num_items
        self.train_U2I = train_U2I
        self.gama = gama

    def to_edge(self):
        train_U, train_I = [], []
        for user_id, items in self.train_U2I.items():
            train_U.extend([user_id] * len(items))
            train_I.extend(items)

        train_U = np.asarray(train_U, dtype=np.int64)
        train_I = np.asarray(train_I, dtype=np.int64)

        row = np.concatenate([train_U, train_I + self.num_users])
        col = np.concatenate([train_I + self.num_users, train_U])
        edge_weight = np.ones_like(row, dtype=np.float32)
        edge_index = np.stack([row, col]).tolist()
        return train_U, train_I, edge_index, edge_weight.tolist()


class LaplaceGraph(Graph):
    def __init__(self, num_users, num_items, train_U2I, gama=0.5):
        super(LaplaceGraph, self).__init__(num_users, num_items, train_U2I, gama)

    def generate(self):
        _, _, edge_index, edge_weight = self.to_edge()
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        edge_index, edge_weight = self.add_self_loop(edge_index, edge_weight)
        edge_index, edge_weight = self.norm(edge_index, edge_weight)
        return self.mat(edge_index, edge_weight)

    def add_self_loop(self, edge_index, edge_weight):
        loop_index = torch.arange(0, self.num_nodes, dtype=torch.long)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        loop_weight = torch.ones(self.num_nodes, dtype=torch.float32)
        edge_index = torch.cat([edge_index, loop_index], dim=-1)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=-1)
        return edge_index, edge_weight

    def norm(self, edge_index, edge_weight):
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(self.num_nodes, dtype=torch.float32)
        deg = deg.scatter_add(0, col, edge_weight)
        deg_inv_sqrt = deg.pow(-1 * self.gama)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index, edge_weight

    @property
    def num_nodes(self):
        return self.num_users + self.num_items

    def mat(self, edge_index, edge_weight):
        return torch.sparse_coo_tensor(
            edge_index,
            edge_weight,
            torch.Size([self.num_nodes, self.num_nodes]),
        ).coalesce()


def unpack_interaction(record):
    if isinstance(record, dict):
        if "user" in record and "item" in record:
            return int(record["user"]), int(record["item"])
        if "u" in record and "i" in record:
            return int(record["u"]), int(record["i"])

    if torch.is_tensor(record):
        flat = record.detach().view(-1).tolist()
        if len(flat) >= 2:
            return int(flat[0]), int(flat[1])

    if isinstance(record, np.ndarray):
        flat = record.reshape(-1)
        if flat.size >= 2:
            return int(flat[0]), int(flat[1])

    if isinstance(record, (list, tuple)) and len(record) >= 2:
        return int(record[0]), int(record[1])

    raise ValueError(f"Unsupported interaction format: {type(record)}")


class PopularityBucketSampler:
    def __init__(self, data, num_groups, balanced_neg_per_group, seed):
        self.num_items = data.num_items
        self.num_groups = max(2, int(num_groups))
        self.balanced_neg_per_group = max(1, int(balanced_neg_per_group))
        self.rng = np.random.default_rng(seed)

        pop_array = np.asarray(data.pop_train_count, dtype=np.float32)
        sorted_items = np.argsort(pop_array, kind="mergesort")
        bucket_splits = np.array_split(sorted_items, self.num_groups)

        self.item_group = np.zeros(self.num_items, dtype=np.int64)
        self.group_items = {}
        self.group_stats = []

        for group_id, item_ids in enumerate(bucket_splits):
            item_ids = np.asarray(item_ids, dtype=np.int64)
            self.group_items[group_id] = item_ids
            if item_ids.size > 0:
                self.item_group[item_ids] = group_id
                self.group_stats.append(
                    {
                        "group_id": group_id,
                        "size": int(item_ids.size),
                        "min_pop": float(pop_array[item_ids].min()),
                        "max_pop": float(pop_array[item_ids].max()),
                        "mean_pop": float(pop_array[item_ids].mean()),
                    }
                )
            else:
                self.group_stats.append(
                    {
                        "group_id": group_id,
                        "size": 0,
                        "min_pop": 0.0,
                        "max_pop": 0.0,
                        "mean_pop": 0.0,
                    }
                )

        self.all_items = np.arange(self.num_items, dtype=np.int64)
        self.user_history = {
            int(user_id): set(int(item_id) for item_id in item_ids)
            for user_id, item_ids in data.train_U2I.items()
        }

    def sample_negative(self, user_id, group_id, extra_forbidden=None, max_trials=64):
        forbidden = set()
        forbidden.update(self.user_history.get(int(user_id), set()))
        if extra_forbidden is not None:
            forbidden.update(int(item_id) for item_id in extra_forbidden)

        candidate_pool = self.group_items.get(int(group_id), self.all_items)
        sampled = self._try_sample_from_pool(candidate_pool, forbidden, max_trials)
        if sampled is not None:
            return sampled

        sampled = self._try_sample_from_pool(self.all_items, forbidden, max_trials * 2)
        if sampled is not None:
            return sampled

        for item_id in self.all_items:
            if int(item_id) not in forbidden:
                return int(item_id)

        raise RuntimeError("Failed to sample a valid negative item.")

    def sample_same_group_negative(self, user_id, pos_item_id, extra_forbidden=None):
        group_id = int(self.item_group[int(pos_item_id)])
        return self.sample_negative(user_id, group_id, extra_forbidden=extra_forbidden)

    def sample_balanced_negatives(self, user_id, pos_item_id, extra_forbidden=None):
        negatives = []
        local_forbidden = set(extra_forbidden or [])
        local_forbidden.add(int(pos_item_id))
        pos_group = int(self.item_group[int(pos_item_id)])

        for group_id in range(self.num_groups):
            if group_id == pos_group:
                continue
            for _ in range(self.balanced_neg_per_group):
                neg_item = self.sample_negative(user_id, group_id, extra_forbidden=local_forbidden)
                negatives.append(neg_item)
                local_forbidden.add(int(neg_item))
        return negatives

    def _try_sample_from_pool(self, pool, forbidden, max_trials):
        if pool.size == 0:
            return None

        for _ in range(max_trials):
            item_id = int(pool[self.rng.integers(0, pool.size)])
            if item_id not in forbidden:
                return item_id
        return None


def next_batch_group_balanced(data, batch_size, sampler):
    indices = np.arange(len(data.training_data))
    sampler.rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batch_indices = indices[start:end]

        users = []
        pos_items = []
        same_neg_items = []
        balanced_neg_items = []
        pos_groups = []

        for idx in batch_indices:
            user_id, pos_item_id = unpack_interaction(data.training_data[idx])
            pos_group = int(sampler.item_group[pos_item_id])

            same_neg_item = sampler.sample_same_group_negative(
                user_id,
                pos_item_id,
                extra_forbidden={pos_item_id},
            )
            balanced_negs = sampler.sample_balanced_negatives(
                user_id,
                pos_item_id,
                extra_forbidden={pos_item_id, same_neg_item},
            )

            users.append(user_id)
            pos_items.append(pos_item_id)
            same_neg_items.append(same_neg_item)
            balanced_neg_items.append(balanced_negs)
            pos_groups.append(pos_group)

        yield (
            np.asarray(users, dtype=np.int64),
            np.asarray(pos_items, dtype=np.int64),
            np.asarray(same_neg_items, dtype=np.int64),
            np.asarray(balanced_neg_items, dtype=np.int64),
            np.asarray(pos_groups, dtype=np.int64),
        )
