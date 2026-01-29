from itertools import product
import multiprocessing
import random
import os
from multiprocessing import Pool
def train_model(params):
    lr, batch_size, dropout_prob, num_emb_list, e_dim, quant_loss_weight, beta, sk_epsilons, kmeans_init, available_gpu = params
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus)  # 设置 GPU ID
    print(f"Running on GPU {available_gpu} with params: {params}")
    # 构建命令行
    cmd = (
        f"CUDA_VISIBLE_DEVICES={str(available_gpu)} " 
        f"python index/main.py --lr {lr} --batch_size {batch_size} --dropout_prob {dropout_prob} "
        f"--num_emb_list {' '.join(map(str, num_emb_list))} --e_dim {e_dim} "
        f"--quant_loss_weight {quant_loss_weight} --beta {beta} "
        f"--sk_epsilons {' '.join(map(str, sk_epsilons))} --kmeans_init {kmeans_init} "
        f"--data_path ./data/ToolBench/toolgen-mean-embeddings-output-sentences-llama2.npy"
    )
    os.system(cmd)
if __name__ == "__main__":
    # 定义超参数搜索范围
    available_gpus = [5,6,7]
    lr_list = [1e-3, 5e-4, 1e-4]
    batch_size_list = [2048, 5096]
    dropout_prob_list = [0.0, 0.1]
    num_emb_list_options = [
        [256, 256, 256], [512, 256, 256], [256, 512, 256], [256, 512, 512],
        [256, 256, 512], [512, 512, 512], [512, 1024, 512], [1024, 512, 512],
        [1024, 1024, 512], [512, 1024, 1024], [1024, 512, 1024], [1024, 1024, 1024]
    ]
    e_dim_list = [32, 64]
    quant_loss_weight_list = [0.5, 1.0, 2.0]
    beta_list = [0.1, 0.25, 0.5]
    sk_epsilons_list = [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [0.1, 0.1, 0.1]]
    kmeans_init_list = [True, False]

    # 随机抽取 20 组超参数组合
    search_space = [
        (
            random.choice(lr_list),
            random.choice(batch_size_list),
            random.choice(dropout_prob_list),
            random.choice(num_emb_list_options),
            random.choice(e_dim_list),
            random.choice(quant_loss_weight_list),
            random.choice(beta_list),
            random.choice(sk_epsilons_list),
            random.choice(kmeans_init_list),
            available_gpus[ _ % len(available_gpus)]  # 动态分配 GPU
        )
        for _ in range(30)  # 控制总实验数量
    ]

    # 控制同时运行的最大进程数为 3
    with Pool(processes=1) as pool:  # 3 表示最大并行进程数
        pool.map(train_model, search_space)