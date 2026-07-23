import torch


class BaseModel(torch.nn.Module):
    def __init__(self, config, data):
        super(BaseModel, self).__init__()

        self.emb_size = config.emb_size
        self.decay = config.decay
        self.layers = config.layers
        self.device = (
            config.device
            if hasattr(config, "device")
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.data = data

        self.num_users = data.num_users
        self.num_items = data.num_items
        self.adj = data.norm_adj.to(self.device)

        self._init_embeddings()

    def _init_embeddings(self):
        user_emb_weight = torch.nn.init.xavier_normal_(
            torch.empty(self.num_users, self.emb_size))
        item_emb_weight = torch.nn.init.xavier_normal_(
            torch.empty(self.num_items, self.emb_size))
        self.user_embeddings = torch.nn.Embedding(
            self.num_users, self.emb_size, _weight=user_emb_weight)
        self.item_embeddings = torch.nn.Embedding(
            self.num_items, self.emb_size, _weight=item_emb_weight)
    def forward(self):
        ego_embeddings = torch.cat(
            [self.user_embeddings.weight, self.item_embeddings.weight], dim=0
        )
        all_emb = [ego_embeddings]

        for _ in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.adj, ego_embeddings)
            all_emb.append(ego_embeddings)

        all_emb = torch.stack(all_emb, dim=1)
        all_emb = torch.mean(all_emb, dim=1)
        user_emb, item_emb = torch.split(all_emb, [self.num_users, self.num_items])
        return user_emb, item_emb

    def inference_embeddings(self):
        return self.forward()

    def full_sort_scores(self, user_indices):
        user_indices = torch.as_tensor(
            user_indices, dtype=torch.long, device=self.device
        )
        user_emb, item_emb = self.inference_embeddings()
        return torch.matmul(user_emb[user_indices], item_emb.t())

    def bpr_loss(self, user_emb, pos_emb, neg_emb, u_idx, i_idx, j_idx):
        pos_score = torch.mul(user_emb, pos_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_emb).sum(dim=1)
        bpr_loss = -torch.log(1e-8 + torch.sigmoid(pos_score - neg_score)).mean()

        user_emb0 = self.user_embeddings.weight[u_idx]
        pos_emb0 = self.item_embeddings.weight[i_idx]
        neg_emb0 = self.item_embeddings.weight[j_idx]
        l2_loss = (
            self.decay
            * 0.5
            * (user_emb0.norm(2).pow(2) + pos_emb0.norm(2).pow(2) + neg_emb0.norm(2).pow(2))
            / float(len(u_idx))
        )

        return bpr_loss, l2_loss

    def batch_loss(self, u_idx, i_idx, j_idx):
        u_idx = torch.as_tensor(u_idx, dtype=torch.long, device=self.device)
        i_idx = torch.as_tensor(i_idx, dtype=torch.long, device=self.device)
        j_idx = torch.as_tensor(j_idx, dtype=torch.long, device=self.device)

        user_embedding, item_embedding = self.forward()
        user_emb = user_embedding[u_idx]
        pos_emb = item_embedding[i_idx]
        neg_emb = item_embedding[j_idx]

        bpr_loss, l2_loss = self.bpr_loss(user_emb, pos_emb, neg_emb, u_idx, i_idx, j_idx)
        batch_loss = bpr_loss + l2_loss

        return batch_loss, bpr_loss, l2_loss
