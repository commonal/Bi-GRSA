import logging
import ast
import numpy as np
import random
import os
import torch
import torch.nn.functional as F


#创建和配置 Python 日志记录器（logger）的工具函数
def get_logger(filename, verbosity=1, name=None):

    """
    创建并配置一个日志记录器，可以同时输出到文件和控制台

    参数:
        filename (str): 日志文件名，会自动添加.txt后缀
        verbosity (int): 日志详细程度，0=DEBUG, 1=INFO, 2=WARNING
        name (str, optional): 日志记录器名称，默认为None

    返回:
        logging.Logger: 配置好的日志记录器实例
    """
    filename = filename + '.txt'  # 确保日志文件名包含.txt后缀
    # 定义不同详细程度对应的日志级别
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # 设置日志格式：[时间戳]日志内容
    formatter = logging.Formatter(
        "[%(asctime)s]%(message)s"
    )
    # 获取或创建日志记录器
    logger = logging.getLogger(name)
    logger.handlers=[]  # 清除已有的处理器
    # 设置日志级别
    logger.setLevel(level_dict[verbosity])
    # 创建并添加文件处理器，输入到文件
    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    #创建并添加控制台处理器,输入到终端显示
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

    
def setup_seed(seed):

    """
    设置随机种子以确保实验结果的可复现性
    参数:
        seed (int): 随机种子值
    该函数通过设置不同库的随机种子，确保每次运行程序时生成的随机数序列相同
    """
    # 设置Python的哈希种子，确保字典等数据结构的迭代顺序一致
    os.environ['PYTHONHASHSEED']=str(seed)


    # 设置Python内置random模块的随机种子
    random.seed(seed)
    # 设置NumPy库的随机种子
    np.random.seed(seed)
    
    # 设置PyTorch的CPU随机种子
    torch.manual_seed(seed)
    # 设置PyTorch的所有GPU设备的随机种子
    torch.cuda.manual_seed_all(seed)
    # 确保CuDNN使用确定性算法，避免非确定性算法带来的结果不一致
    torch.backends.cudnn.deterministic = True


def normalize_device(device_value):
    device_str = str(device_value)
    if device_str.isdigit():
        return f"cuda:{device_str}" if torch.cuda.is_available() else "cpu"
    return device_str


def parse_search_values(raw_value):
    parsed = ast.literal_eval(raw_value)
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]

def resolve_device(device):
    """Accept int / str / torch.device and return a safe torch.device."""
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        if torch.cuda.is_available():
            return torch.device(f'cuda:{device}')
        return torch.device('cpu')
    return torch.device(device)


def primary_topk(topks):
    """Parse the first top-k value from CLI args."""
    if isinstance(topks, str):
        topks = ast.literal_eval(topks)
    if isinstance(topks, (list, tuple)):
        if not topks:
            raise ValueError("topks must not be empty")
        return int(topks[0])
    return int(topks)



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,logger, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.logger=logger

    def __call__(self, val_loss, model,epoch):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            #torch.save(model.state_dict(), self.path + 'best_val_epoch_'+str(self.counter)+'_epoch_'+str(epoch)+'.pt')
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        # if epoch%10==0:
        #     torch.save(model.state_dict(), self.path + 'epoch'+str(epoch)+'.pt')


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path + '/best_val_epoch.pt')
        torch.save(model.state_dict(), self.path + '/the_final_epoch.pt')
        self.val_loss_min = val_loss

# def split_bacth_items(items,popular):
#     G1,G2=[],[]
#     items_sorted=list(np.array(items)[np.argsort(np.array(popular)[items])])
#     num=int(len(items_sorted)/2)
#     G1.extend(items_sorted[0:num])
#     G2.extend(items_sorted[num:])
#     return np.array(G1),np.array(G2)

def split_bacth_items(items, popular, ratio=0.5):
    """
    ratio: 前ratio比例为一组，后(1-ratio)为另一组
    ratio=0.5 → 原版中位数对半分
    ratio=0.3 → top 30%为G1，bottom 70%为G2
    ratio=0.7 → top 70%为G1，bottom 30%为G2
    """
    G1, G2 = [], []
    items_sorted = list(np.array(items)[np.argsort(np.array(popular)[items])])
    num = max(1, int(len(items_sorted) * ratio))  # 原来是硬编码/2
    G1.extend(items_sorted[0:num])
    G2.extend(items_sorted[num:])
    return np.array(G1), np.array(G2)

def alignment_user(x, y): #对齐x和ｙ
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(2).mean()

def knn_alignment(x, y, k=5):
    """
    使用KNN进行对齐，使得x中的每个向量与y中最近的k个向量对齐
    """
    # 确保输入是numpy数组
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    
    # 使用sklearn的NearestNeighbors找到最近邻
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(y)
    distances, indices = nbrs.kneighbors(x)
    
    # 计算对齐损失，使用距离作为损失
    alignment_loss = np.mean(distances)
    
    return torch.tensor(alignment_loss, dtype=torch.float32)

def weighted_knn_alignment(x, y, x_weights, y_weights, k=5):
    """
    加权KNN对齐，根据物品流行度给予不同权重
    x_weights: x中每个向量的权重（如流行度）
    y_weights: y中每个向量的权重（如流行度）
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(x_weights, torch.Tensor):
        x_weights = x_weights.detach().cpu().numpy()
    if isinstance(y_weights, torch.Tensor):
        y_weights = y_weights.detach().cpu().numpy()
    
    # 使用sklearn的NearestNeighbors找到最近邻
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(y)
    distances, indices = nbrs.kneighbors(x)
    
    # 计算加权对齐损失
    weighted_distances = []
    for i in range(len(x)):
        # 对每个x中的向量，计算其与k个最近邻的加权距离
        neighbor_weights = y_weights[indices[i]]
        # 使用流行度权重调整距离，热门物品给予更高权重
        weights = (x_weights[i] + neighbor_weights) / 2  # 平均权重
        weighted_dist = distances[i] * weights
        weighted_distances.append(np.mean(weighted_dist))
    
    alignment_loss = np.mean(weighted_distances)
    return torch.tensor(alignment_loss, dtype=torch.float32)

def asymmetric_knn_alignment(x, y, k=5, temperature=1.0):
    """
    不对称KNN对齐
    x向y对齐，但不强制y向x对齐
    temperature: 温度参数，控制分布的平滑程度
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    
    # 使用sklearn的NearestNeighbors找到最近邻
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(y)
    distances, indices = nbrs.kneighbors(x)
    
    # 应用温度参数
    similarities = 1 - distances  # 转换为相似度
    weighted_similarities = np.exp(similarities / temperature)
    normalized_weights = weighted_similarities / np.sum(weighted_similarities, axis=1, keepdims=True)
    
    # 计算基于softmax权重的对齐损失
    alignment_loss = -np.mean(np.log(normalized_weights + 1e-8))  # 添加小值避免log(0)
    
    return torch.tensor(alignment_loss, dtype=torch.float32)
