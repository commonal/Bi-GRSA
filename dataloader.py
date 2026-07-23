import math
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

import utils

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


def _sample_unobserved_item(num_items, observed_items, max_trials=100):
    observed_items = set(observed_items)
    if len(observed_items) >= num_items:
        raise RuntimeError("Failed to sample a negative item because the user interacted with all items.")

    for _ in range(max_trials):
        candidate = random.randrange(num_items)
        if candidate not in observed_items:
            return candidate

    for candidate in range(num_items):
        if candidate not in observed_items:
            return candidate

    raise RuntimeError("Failed to sample a valid negative item.")


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

        percentile = getattr(config, "global_group_ratio", 50)
        user_interactions = np.asarray(self.active_train_count, dtype=np.float32)
        item_interactions = np.asarray(self.pop_train_count, dtype=np.float32)

        user_threshold = np.percentile(user_interactions, percentile)
        item_threshold = np.percentile(item_interactions, percentile)

        self.active_user_ids = set(np.where(user_interactions >= user_threshold)[0])
        self.popular_item_ids = set(np.where(item_interactions >= item_threshold)[0])
        self.cold_item_ids = set(np.where(item_interactions < item_threshold)[0])

        self.logger.info(
            "user activity threshold: %.4f (percentile=%s)",
            user_threshold,
            percentile,
        )
        self.logger.info(
            "item popularity threshold: %.4f (percentile=%s)",
            item_threshold,
            percentile,
        )

    def analyze_active_user_behavior(self):
        active_users = self.active_user_ids
        active_popular_interactions = 0
        active_unpopular_interactions = 0

        for user in active_users:
            if user not in self.train_U2I:
                continue
            for item in self.train_U2I[user]:
                if item in self.popular_item_ids:
                    active_popular_interactions += 1
                else:
                    active_unpopular_interactions += 1

        total_active_interactions = active_popular_interactions + active_unpopular_interactions
        if total_active_interactions == 0:
            return 0, 0

        self.logger.info(
            "active users -> popular items interactions: %d",
            active_popular_interactions,
        )
        self.logger.info(
            "active users -> unpopular items interactions: %d",
            active_unpopular_interactions,
        )
        self.logger.info(
            "active users total interactions: %d",
            total_active_interactions,
        )
        self.logger.info(
            "active users -> popular items ratio: %.2f%%",
            active_popular_interactions / total_active_interactions * 100,
        )
        self.logger.info(
            "active users -> unpopular items ratio: %.2f%%",
            active_unpopular_interactions / total_active_interactions * 100,
        )
        return active_popular_interactions, active_unpopular_interactions

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

    def bc_loss_data(self):
        pop_user = {key: len(value) for key, value in self.train_U2I.items()}
        pop_item = {key: len(value) for key, value in self.train_I2U.items()}

        sorted_pop_user = sorted(set(pop_user.values()))
        sorted_pop_item = sorted(set(pop_item.values()))
        self.n_user_pop = len(sorted_pop_user)
        self.n_item_pop = len(sorted_pop_item)

        user_idx = {value: idx for idx, value in enumerate(sorted_pop_user)}
        item_idx = {value: idx for idx, value in enumerate(sorted_pop_item)}

        self.user_pop_idx = np.zeros(self.num_users, dtype=int)
        self.item_pop_idx = np.zeros(self.num_items, dtype=int)
        for key, value in pop_user.items():
            self.user_pop_idx[key] = user_idx[value]
        for key, value in pop_item.items():
            self.item_pop_idx[key] = item_idx[value]

        threshold_map = {
            "ml-1m": 50,
            "yelp2018": 47,
            "douban-book": 22,
            "gowalla": 22,
            "amazon-book": 33,
            "ml-20m": 100,
            "addressa": 30,
        }
        dataset_key = self.dataset_name.replace(".new", "")
        split_value = threshold_map.get(dataset_key)
        if split_value is None:
            split_value = float(
                np.percentile(np.asarray(self.pop_train_count, dtype=np.float32), 50)
            )

        unpopular_item = []
        popular_item = []
        for item, pop in enumerate(self.pop_train_count):
            if pop <= split_value:
                unpopular_item.append(item)
            else:
                popular_item.append(item)
        return unpopular_item, popular_item


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


def next_batch_pairwise(data, batch_size):
    training_data = data.training_data
    random.shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)

    while batch_id < data_size:
        batch_end = min(batch_id + batch_size, data_size)
        users = [training_data[idx][0] for idx in range(batch_id, batch_end)]
        items = [training_data[idx][1] for idx in range(batch_id, batch_end)]
        batch_id = batch_end

        u_idx, i_idx, j_idx = [], [], []
        for index, user in enumerate(users):
            u_idx.append(user)
            i_idx.append(items[index])
            j_idx.append(_sample_unobserved_item(data.num_items, data.train_U2I[user]))
        yield u_idx, i_idx, j_idx


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


class CrossPairwiseSampler:
    def __init__(
            self,
            data,
            batch_size,
            max_k_interact=3,
            sample_ratio=1.0,
            sample_rate=1.0,
            seed=2026,
            group_retry=80,
            inner_retry=120,
    ):
        self.data = data
        self.batch_size = int(batch_size)
        self.max_k_interact = max(2, int(max_k_interact))
        self.sample_ratio = float(sample_ratio)
        self.sample_rate = float(sample_rate)
        self.group_retry = max(1, int(group_retry))
        self.inner_retry = max(1, int(inner_retry))

        self.rng = np.random.default_rng(seed)
        self.training_pairs = np.asarray(data.training_data, dtype=np.int64)
        self.num_pairs = int(len(self.training_pairs))
        self.user_history = {
            int(user_id): set(int(item_id) for item_id in item_ids)
            for user_id, item_ids in data.train_U2I.items()
        }
        self.steps_per_epoch = max(1, math.ceil(self.num_pairs / max(1, self.batch_size)))
        self.k_values = list(range(2, self.max_k_interact + 1))
        self.group_plan = self._build_group_plan()

    def _build_group_plan(self):
        if self.max_k_interact == 2:
            return {2: max(1, int(math.ceil(self.batch_size * self.sample_rate)))}

        ratios = np.power(
            self.sample_ratio,
            np.arange(self.max_k_interact - 2, -1, -1, dtype=np.float32),
        )
        base_counts = np.round(self.batch_size * ratios / ratios.sum()).astype(np.int64)
        base_counts = np.maximum(base_counts, 1)
        scaled_counts = np.ceil(base_counts * self.sample_rate).astype(np.int64)
        return {
            k_value: int(max(1, scaled_counts[offset]))
            for offset, k_value in enumerate(self.k_values)
        }

    def __len__(self):
        return self.steps_per_epoch

    def _sample_interaction(self):
        pair = self.training_pairs[self.rng.integers(0, self.num_pairs)]
        return int(pair[0]), int(pair[1])

    def _can_append(self, current_users, current_items, cand_user, cand_item):
        if cand_user in current_users or cand_item in current_items:
            return False

        cand_history = self.user_history.get(cand_user, set())
        if any(existing_item in cand_history for existing_item in current_items):
            return False

        for existing_user in current_users:
            if cand_item in self.user_history.get(existing_user, set()):
                return False

        return True

    def _sample_one_group(self, k_value):
        for _ in range(self.group_retry):
            start_user, start_item = self._sample_interaction()
            group_users = [start_user]
            group_items = [start_item]

            for _ in range(self.inner_retry):
                if len(group_users) == k_value:
                    return group_users, group_items

                cand_user, cand_item = self._sample_interaction()
                if self._can_append(group_users, group_items, cand_user, cand_item):
                    group_users.append(cand_user)
                    group_items.append(cand_item)

            if len(group_users) == k_value:
                return group_users, group_items

        return None

    def sample_batch(self):
        sampled = {}
        for k_value, num_groups in self.group_plan.items():
            user_groups = []
            item_groups = []
            attempts = 0
            max_attempts = max(num_groups * 4, 16)

            while len(user_groups) < num_groups and attempts < max_attempts:
                group = self._sample_one_group(k_value)
                attempts += 1
                if group is None:
                    continue
                group_users, group_items = group
                user_groups.append(group_users)
                item_groups.append(group_items)

            if user_groups:
                sampled[k_value] = (
                    np.asarray(user_groups, dtype=np.int64),
                    np.asarray(item_groups, dtype=np.int64),
                )

        if not sampled:
            raise RuntimeError(
                "CrossPairwiseSampler failed to construct any valid CPR groups. "
                "You can try reducing cpr_k or sample_rate."
            )
        return sampled

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            yield self.sample_batch()


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


def user_items_2_group_pop(data):
    group_1, group_2 = [], []
    for user_id in data.train_U2I.keys():
        items = data.train_U2I[user_id]
        items_sorted = list(
            np.asarray(items)[np.argsort(np.asarray(data.pop_train_count)[items])]
        )
        if len(items_sorted) % 2 != 0:
            drop_idx = random.sample(range(len(items_sorted)), 1)[0]
            items_sorted = np.delete(items_sorted, drop_idx)
        half = int(len(items_sorted) / 2)
        group_1.extend(items_sorted[0:half])
        group_2.extend(items_sorted[half:])
    return np.asarray(group_1), np.asarray(group_2)


def item_users_2_group_act(data, min_users=1):
    group_1, group_2 = [], []
    for item_id in data.train_I2U.keys():
        users = data.train_I2U[item_id]
        if len(users) < min_users:
            continue

        users_sorted = list(
            np.asarray(users)[np.argsort(np.asarray(data.active_train_count)[users])]
        )
        if len(users_sorted) % 2 != 0:
            users_sorted.pop(random.choice(range(len(users_sorted))))
        half = len(users_sorted) // 2
        group_1.extend(users_sorted[:half])
        group_2.extend(users_sorted[half:])
    return np.asarray(group_1), np.asarray(group_2)


def save_groups(G1, G2, G3, G4, dataset_name="default", cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.join(cache_dir, f"{dataset_name}_saved_groups.npz")
    np.savez(filename, G1=G1, G2=G2, G3=G3, G4=G4)


def load_groups(dataset_name="default", cache_dir="cache"):
    filename = os.path.join(cache_dir, f"{dataset_name}_saved_groups.npz")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No cached file found for dataset: {dataset_name}")
    loaded = np.load(filename)
    return loaded["G1"], loaded["G2"], loaded["G3"], loaded["G4"]


def get_groups(data, force_reload=False, cache_dir="cache"):
    dataset_name = data.dataset_name
    filename = os.path.join(cache_dir, f"{dataset_name}_saved_groups.npz")

    if not force_reload and os.path.exists(filename):
        print(f"Loading cached group data for dataset: {dataset_name}...")
        return load_groups(dataset_name, cache_dir)

    print(f"Generating new group data for dataset: {dataset_name}...")
    G1, G2 = user_items_2_group_pop(data)
    G3, G4 = item_users_2_group_act(data)
    save_groups(G1, G2, G3, G4, dataset_name, cache_dir)
    return G1, G2, G3, G4


class Config:
    dataset_name: str = "douban"
    dataset_path: str = "OOD_Data"
    bpr_num_neg: int = 1


if __name__ == "__main__":
    config = Config()
    filename = f"{config.dataset_name}.log"
    logger = utils.get_logger(filename)
    data = Data(config, logger)
