"""

背景：
    基于手机的20个特征->预测手机的价格区间（4个）

ann案例的实现步骤：
    1.构建数据集
    2.搭建神经网络
    3.模型训练
    4.模型测试

优化思路：
    1.SGD->Adam
    2.0.001->0.0001
    3.BN 批量归一化处理
    4.加入GPU加速
"""

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from torchsummary import summary
from sklearn.preprocessing import StandardScaler

# ===================== 【核心：自动选择GPU】 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 当前使用设备：", device)

# todo 1.定义函数 购机数据集
def create_dataset():
    # 1.加载数据集
    data = pd.read_csv('手机价格预测.csv')

    # 2.获取x特征 y标签列
    x,y = data.iloc[:,:-1],data.iloc[:,-1]

    # 3.把特征列转化成浮点型
    x = x.astype(np.float32)

    #归一化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # 4.切分训练集和测试级
    x_train,x_test,y_train,y_test = train_test_split(
        x,y,test_size=0.2,random_state=3,stratify=y
    )

    # 5.把数据集封装成张量数据集
    train_dataset = TensorDataset(torch.tensor(x_train),torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test),torch.tensor(y_test.values))

    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))


# todo 2.搭建神经网络
class PhonePriceModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.output(x)
        return x


# todo 3.模型训练
def train(train_dataset,input_dim,ouput_dim):
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle = True)

    # ===================== 修改1：模型放到GPU =====================
    model = PhonePriceModel(input_dim,ouput_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = 0.001)
    epochs = 100

    for epoch in range(epochs):
        total_loss , batch_num = 0.0,0
        start = time.time()

        for x,y in train_loader:
            model.train()

            # ===================== 修改2：数据放到GPU =====================
            x, y = x.to(device), y.to(device)

            y_pred= model(x)
            loss = criterion(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_num += 1

        print(f'epoch:{epoch+1},loss:{total_loss/batch_num},time:{time.time()-start}')

    torch.save(model.state_dict(), 'phone.pth')


# todo 4.模型测试
def evaluate(test_dataset,input_dim,ouput_dim):
    # ===================== 修改3：模型放到GPU =====================
    model = PhonePriceModel(input_dim,ouput_dim).to(device)

    model.load_state_dict(torch.load('phone.pth'))
    test_loader = DataLoader(test_dataset,batch_size=8,shuffle=False)
    correct = 0

    model.eval()
    with torch.no_grad():  # 测试时关闭梯度计算
        for x,y in test_loader:
            # ===================== 修改4：数据放到GPU =====================
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            y_pred = torch.argmax(y_pred,dim=1)

            correct += (y_pred == y).sum().item()

    print(f'准确率：{correct/len(test_dataset) * 100:.2f}%')


if __name__ == '__main__':
    train_dataset ,test_dataset,input_dim,ouput_dim = create_dataset()
    train(train_dataset, input_dim, ouput_dim)
    evaluate(test_dataset,input_dim,ouput_dim)