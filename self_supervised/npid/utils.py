import math

import torch
from torch.autograd import Function
from torch import nn

from rich import print

class NCEFunction(Function):
    @staticmethod
    def forward(self, x, y, memory, idx, params):
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()

        momentum = params[3].item()
        batch_size = x.size(0)
        outputSize = memory.size(0)
        in_dim = memory.size(1)

        # sample positives & negatives
        # idx.select(1, 0).copy_(y.data)
        idx[:, 0] = y  # positive sample
        # sample correspoinding weights
        # weight = torch.index_select(memory, 0, idx.view(-1))
        weight = memory[idx.view(-1)].view(batch_size, K, in_dim)
        # weight.resize_(batch_size, K, in_dim)

        # inner product
        # with torch.no_grad():
        out = (torch.einsum("bki, bi -> bk", weight, x) / T).exp()
            # out = torch.bmm(weight, x.resize_(batch_size, in_dim, 1))
            # out.div_(T).exp_()  # batch_size * self.K+1
            # x.resize_(batch_size, in_dim)

        if Z < 0:
            params[2] = out.mean() * outputSize
            Z = params[2].item()
            print("normalization constant Z is set to {:.1f}".format(Z))

        # out.div_(Z).resize_(batch_size, K)
        out /= Z

        self.save_for_backward(x, memory, y, weight, out, params)

        return out

    @staticmethod
    def backward(self, grad_output):
        x, memory, y, weight, out, params = self.saved_tensors
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        momentum = params[3].item()
        # batch_size = grad_output.size(0)

        # gradients d Pm / d linear = exp(linear) / Z
        # grad_output.data.mul_(out.data)
        grad_output = grad_output * out / T
        # add temperature
        # grad_output.data.div_(T)

        # grad_output.data.resize_(batch_size, 1, K)

        # gradient of linear
        # grad_input = torch.bmm(grad_output.data, weight)
        grad_input = torch.einsum('bk, bki -> bi', grad_output, weight)
        # grad_input.resize_as_(x)

        # update the non-parametric data
        weight_pos = weight.select(1, 0).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return grad_input, None, None, None, None


class NCEAverage(nn.Module):
    """
    memory: [length_train x feature_dim]
    idx: [batch_size x K]
    """
    def __init__(self, in_dim, out_dim, K, T=0.07, momentum=0.5):
        super().__init__()
        self.out_dim = out_dim
        self.unigrams = torch.ones(self.out_dim)
        self.multinomial = AliasMethod(self.unigrams)
        self.K = K

        self.register_buffer("params", torch.tensor([K, T, -1, momentum]))
        stdv = 1.0 / math.sqrt(in_dim / 3)
        self.register_buffer(
            "memory", torch.rand(out_dim, in_dim).mul_(2 * stdv).add_(-stdv)
        )

    def forward(self, x, y):
        # batch_size = x.size(0)
        batch_size = x.shape[0]
        idx = self.multinomial.draw(batch_size * self.K).view(batch_size, -1)  # [batch_size, K]
        out = NCEFunction.apply(x, y, self.memory, idx.to(x.device), self.params)
        return out


class AliasMethod(object):
    '''
        From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def to(self, device): 
        self.prob = self.prob.to(device)
        self.alias = self.alias.to(device)

    def draw(self, N):
        '''
            Draw N samples from multinomial
        '''
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj


class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        T = params[0].item()
        batchSize = x.size(0)

        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(T)  # batchSize * N

        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, grad_output):
        x, memory, y, params = self.saved_tensors
        batchSize = grad_output.size(0)
        T = params[0].item()
        momentum = params[1].item()

        # add temperature
        grad_output.data.div_(T)

        # gradient of linear
        grad_input = torch.mm(grad_output.data, memory)
        grad_input.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return grad_input, None, None, None


class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.07, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer("params", torch.tensor([T, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer(
            "memory", torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        )

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out
