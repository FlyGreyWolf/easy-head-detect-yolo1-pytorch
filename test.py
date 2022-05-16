import torch.nn as nn
import torch
from main import test

if __name__ == '__main__':
    test()

#
# print(10.2%3)
# loss = nn.MSELoss(reduction='sum')
# # input = torch.randn(3, 5, requires_grad=True)
# # target = torch.randn(3, 5)
# input = torch.tensor([1.0,2.0,3.0],requires_grad=True)
# target = torch.tensor([4.0,2.0,5.0])
# output = loss(input, target)
# print(output)

# a = [(1,2,3,4), (6,7,8,9)]
#
# b = torch.tensor(a)
# print(b)
#
# class CustomLoss(nn.Module):
#     def __init__(self):
#         super(CustomLoss, self).__init__()
#
#     def forward(self, x, y):
#         my_loss = torch.mean(x-y) # x与y相减后平方，求均值即为MSE
#         return my_loss
#
#
# model_loss = CustomLoss()  # 自定义loss
# a = torch.tensor([1.0,2.0,3.0])
# b = torch.tensor([4.0,5.0,6.0])
# c = model_loss(a,b)
# print(c)
#
# a = torch.tensor([[1.0, 0.0],
#         [0.0, 1.0]])
#
# b = torch.tensor([[1.0, 0.0],
#         [1.0, 1.0]])
#
# mse_loss = nn.MSELoss()
# c = mse_loss(a,b)
# print(c)