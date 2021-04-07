# Graph Neural Network

## Pytorch Geometric

### Message Passing

$$
\boldsymbol{x}_i^{(k)} = \gamma^{(k-1)}\left(\boldsymbol{x}_i^{(k-1)}, \Box_{j \in \mathcal{N}(i)}\phi^{(k)}\left(\boldsymbol{x}_i^{(k-1)}, \boldsymbol{x}_j^{(k-1)}, e_{j,i}\right) \right)
$$

These functions correspond to different methods.

- $\phi$ - `message()` and `propagate()`
- $\Box$ - `aggregate()`
- $\gamma$ - `update()`

### GCNConv

$$
\mathbf{x}_{i}^{(k)}=\sum_{j \in \mathcal{N}(i) \cup\{i\}} \frac{1}{\sqrt{\operatorname{deg}(i)} \cdot \sqrt{\operatorname{deg}(j)}} \cdot\left(\boldsymbol{\Theta} \cdot \mathbf{x}_{j}^{(k-1)}\right)
$$

- $\phi$ - MLP and normalization
- $\Box$ - "add"
- $\gamma$ - Identity

### GAT

#### Attention

$$
\text{Attention}(i,j) = \text{softmax}_j\left(e_{i, j}\right)=\frac{\exp \left(e_{i, j}\right)}{\sum_{k \in N(i)} \exp \left(e_{i, k}\right)}
$$

$$
e_{i,j} = \text{LeakyReLU}\left(\boldsymbol{\Theta}\left[\mathbf{W} h_{i} \| \mathbf{W} h_{j}\right]\right)
$$

where $\|$ means concatenation and $\boldsymbol{\Theta}_{2 \times in\_dim, 1}$ is a linear projection that takes in a pair of node features and output a scalar.

#### Single-head attention

$$
h_{i}^{\prime}=\sigma\left(\sum_{j \in N(i)} \alpha_{i, j} \mathbf{W} h_{j}\right)
$$

where $\sigma$ is the ELU activation.

#### Multi-head attention

Concatenate output of different heads

$$
h_{i}^{\prime}=\|_{k=1}^{K} \sigma\left(\sum_{j \in N(i)} \alpha_{i, j}^{k} \mathbf{W}^{k} h_{j}\right)
$$

or average them

$$
h_{i}^{\prime}=\sigma\left(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in N(i)} \alpha_{i, j}^{k} \mathbf{W}^{k} h_{j}\right)
$$

According to the original paper, they use concatenation in hidden layers and average in the final layer, for more stable results.
