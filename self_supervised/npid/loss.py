import torch
from torch import nn


eps = 1e-7


class NCECriterion(nn.Module):
    def __init__(self, num_data):
        super(NCECriterion, self).__init__()
        self.num_data = num_data

    def forward(self, x, targets):
        K = x.shape[1] - 1  # number of negative samples
        # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
        lnPmt = x[:, 0] / (x[:, 0] + K / self.num_data + eps)
        # eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
        lnPon = (K / self.num_data) / (x[:, 1:] + K / self.num_data + eps)
        # equation 6 in ref. A
        lnPmtsum = lnPmt.log().mean()
        lnPonsum = lnPon.log().sum(dim=1).mean()
        loss = -(lnPmtsum + lnPonsum)
        # loss = -(lnPmtsum + lnPonsum) / batchSize
        return loss


def test():
    criterion = NCECriterion(40000)
    preds = torch.rand(10, 400)
    loss = criterion(preds, 0)
    print(loss)


if __name__ == '__main__':
    from rich import print
    test()
