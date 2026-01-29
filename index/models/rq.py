import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 0.25,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100, graph_lambda = 10):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        beta=self.beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters,graph_lambda = graph_lambda)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, sub_sim = None, use_sk=True):
        all_losses = []
        all_indices = []
        if sub_sim != None:
            all_graph_losses = []
            all_commitment_losses = []
            all_codebook_losses = []

        x_q = 0
        residual = x # batch_size * embedding_dim
        for quantizer in self.vq_layers:
            if sub_sim != None:
                x_res, loss, indices, graph_loss, commitment_loss, codebook_loss = quantizer(residual, sub_sim, use_sk=use_sk)
            else:
                x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res # batchsize * embedding_dim
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)
            if sub_sim != None:
                all_graph_losses.append(graph_loss)
                all_commitment_losses.append(commitment_loss)
                all_codebook_losses.append(codebook_loss)
                mean_graph_loss = torch.stack(all_graph_losses).mean()
                mean_commitment_loss = torch.stack(all_commitment_losses).mean()
                mean_codebook_loss = torch.stack(all_codebook_losses).mean()

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        if sub_sim != None:
            return x_q, mean_losses, all_indices, mean_graph_loss, mean_commitment_loss, mean_codebook_loss
        else:
            return x_q, mean_losses, all_indices