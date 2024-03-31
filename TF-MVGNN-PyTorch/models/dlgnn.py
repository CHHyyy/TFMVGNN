import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F



class DLGNNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0, K: int = 2):
        super(DLGNNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.K = K
        self.register_buffer("adj", adj)
        # 初始化正向和反向权重矩阵，每个k都有一组权重
        self.E1 = nn.Parameter(torch.FloatTensor(358, 10))
        self.E2 = nn.Parameter(torch.FloatTensor(358, 10))
        self.weights_f = nn.ParameterList([nn.Parameter(torch.FloatTensor(self._num_gru_units + 1, self._output_dim)) for _ in range(K+1)])
        self.weights_b = nn.ParameterList([nn.Parameter(torch.FloatTensor(self._num_gru_units + 1, self._output_dim)) for _ in range(K+1)])
        self.weights_adp = nn.Parameter(torch.FloatTensor(num_gru_units + 1, output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.E1)
        nn.init.xavier_uniform_(self.E2)
        for weights in self.weights_f:
            nn.init.xavier_uniform_(weights)
        for weights in self.weights_b:
            nn.init.xavier_uniform_(weights)
        nn.init.xavier_uniform_(self.weights_adp)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        hidden_state = hidden_state.reshape((batch_size, num_nodes, self._num_gru_units))
        concatenation = torch.cat((inputs, hidden_state), dim=2)

        adj = self.adj
        rowsum = adj.sum(dim=1)
        epsilon = 1e-10
        P_f = adj / (rowsum.unsqueeze(1) + epsilon)   # Forward transition matrix
        rowsum_T = adj.transpose(-1, -2).sum(dim=1)
        P_b = adj.transpose(-1, -2) / (rowsum_T.unsqueeze(1) + epsilon) # Backward transition matrix
        E1E2T = torch.matmul(self.E1, self.E2.T)
        A_adp = F.softmax(F.relu(E1E2T), dim=1)

        outputs = torch.zeros(batch_size, num_nodes, self._output_dim).to(inputs.device)

        for k in range(self.K + 1):
            T_k_forward = torch.matrix_power(P_f, k)
            T_k_backward = torch.matrix_power(P_b, k)
            T_k_adaptive = torch.matrix_power(A_adp, k)

            W_k_f = self.weights_f[k]
            W_k_b = self.weights_b[k]

            outputs += T_k_forward.matmul(concatenation).matmul(W_k_f)
            outputs += T_k_backward.matmul(concatenation).matmul(W_k_b)
            outputs += T_k_adaptive.matmul(concatenation).matmul(self.weights_adp)

        outputs += self.biases  # Add biases
        outputs = F.relu(outputs)  # 在这里应用ReLU激活函数
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
            "K": self.K,
        }


class DLGNNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(DLGNNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", adj)
        # Apply thresholding to the adjacency matrix
        self.graph_conv1 = DLGNNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim*2, bias=1.0
        )
        self.graph_conv2 = DLGNNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class DLGNN(nn.Module):
    def __init__(self, adj, hidden_dim: int, **kwargs):
        super(DLGNN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.adj = torch.where(self.adj >= 0.6, torch.tensor(1.0, device=self.adj.device),
                               torch.tensor(0.0, device=self.adj.device))
        self.dlgnn_cell = DLGNNCell(self.adj, self._input_dim, self._hidden_dim)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}