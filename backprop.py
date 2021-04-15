'''
YouTube videos:
https://www.youtube.com/watch?v=DbeIqrwb_dE by Python Engineer
https://www.youtube.com/watch?v=3Kb0QS6z7WA by Python Engineer
https://www.youtube.com/watch?v=E-I2DNVzQLg by Python Engineer
[What is Automatic Differentiation?](https://www.youtube.com/watch?v=wG_nF1awSSY) by Ari Seff

'''

from numpy.core.numeric import True_
import torch


def example_0():
    # Goal: y = 2 * x
    X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

    w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    # model prediction, this is normally our neural network
    def forward(x):
        return w * x


    # loss: MSE
    def loss(y, y_predicted):
        return ((y_predicted - y) ** 2).mean()

    # Training
    lr = 0.01
    # We need a relatively large epoch because autograd is not as exact as the numerical gradient.
    epoch = 60

    for e in range(epoch):
        y_pred = forward(X)
        l = loss(Y, y_pred)
        l.backward()
        # update weights. This step should not be part of our computation graph
        with torch.no_grad():
            w -= lr * w.grad
        # zero gradients
        w.grad.zero_()
        if e % 10 == 0:
            print(f'epoch {e}: w = {w:.3f}, loss = {l:.6f}')

    print(f'Prediction after training: f(5) = {forward(5):.3f}')


def example1():
    '''
    Three ways to prevent being tracked by the computation graph.
    - x.require_grad_(False) / x.require_grad = False
    - x.detach()
    - with torch.no_grad() 
    '''
    x = torch.randn(3, requires_grad=True)
    print(x)

    y = x * 2
    print(y)

    with torch.no_grad():
        y = x * 2
        print(y)


def example3():
    '''
    Multiple backprop will add gradients, so that you can do gradient accumulation. But also remember to call zero_grad() on your optimizer.
    '''
    weights = torch.ones(4, requires_grad=True)
    for epoch in range(3):
        model_output = (weights + 3).sum()
        model_output.backward()
        print(weights.grad)


def example4():
    '''
    Directional derivative (compute Jacobian-vector product without ever computing the Jacobian matrix itself)
    '''
    x = torch.tensor([1.123, 3.21, 3.12], requires_grad=True)
    print(x)
    y = x + 2
    print(y)
    z = y * y * 2
    # z = z.mean()
    print(z)
    v = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)
    # v = torch.tensor([1, 1, 1], dtype=torch.float32)
    z.backward(v)
    print(x.grad)
    

if __name__ == '__main__':
    example4()

