import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from load_data import load_EOD_data, load_relation_data
from evaluator import evaluate
import numpy as np
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch.optim as optim
from tqdm import tqdm
import copy

seed = 123456789
np.random.seed(seed)


class GraphNet(nn.Module):
    def __init__(self):
        super(GraphNet, self).__init__()
        self.conv1 = GCNConv(64, 32)  # first skip through conv1
        self.shortcut1 = nn.Conv1d(64, 32, kernel_size=1, stride=1, bias=False)

        self.conv2 = GCNConv(32, 16)

        self.conv3 = GCNConv(16, 16)  # second skip through conv3
        self.shortcut2 = nn.Conv1d(16, 16, kernel_size=1, stride=1, bias=False)

        self.conv4 = GCNConv(16, 16)

        self.fc1 = nn.Linear(1026 * 16, 4096)
        self.fc2 = nn.Linear(4096, 1026)

    # x represents our data
    def forward(self, x, edge_index):
        # Flatten x with start_dim=1
        dim0, dim1, dim2 = x.shape[0], x.shape[1], x.shape[2]  # [16, 1026, 64]
        input = x.reshape(dim0 * dim1, -1, 1)
        x = x.reshape(dim0 * dim1, -1)  # [16*1026, 64]
        # print('input shape = ', input.shape) #(1026, 16*64)
        # print('edge_index shape = ', edge_index.shape)
        x = F.relu(self.conv1(x, edge_index))  # [16*1026, 32]
        x = F.dropout(x, training=self.training)
        # print('x shape = ', x.shape) #(1026, 512)
        skip = self.shortcut1(input).reshape(dim0 * dim1, -1)  # [dim0*dim1, 32]
        x2 = self.conv2(x + skip, edge_index)
        x2 = F.relu(x2)  # [16*1026, 16]
        # print('x2 = ', x2.shape)
        x3 = F.relu(self.conv3(x2, edge_index))
        skip2 = self.shortcut2(x2.reshape(dim0 * dim1, -1, 1)).reshape(dim0 * dim1, -1)

        x4 = F.relu(self.conv4(x3 + skip2, edge_index).reshape(dim0, -1))
        # print(x.shape)
        x = torch.flatten(x4, 1)  # [16,1026*32]
        # print(x.shape)
        # Pass data through fc1
        # x = self.fc1(x)
        # print(x.shape)
        # x = F.relu(x)
        # print(x.shape)
        x = self.fc1(x)
        pred = self.fc2(x).reshape(dim0, -1)  # [16,1026]

        return pred, x4


def load_EOD_data(data_path, market_name, tickers, steps=1):
    eod_data = []
    masks = []
    ground_truth = []
    base_price = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ':
            # remove the last day since lots of missing data
            single_EOD = single_EOD[:-1, :]
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0],
                                 single_EOD.shape[1] - 1], dtype=np.float32)
            masks = np.ones([len(tickers), single_EOD.shape[0]],
                            dtype=np.float32)
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
                                    dtype=np.float32)
            base_price = np.zeros([len(tickers), single_EOD.shape[0]],
                                  dtype=np.float32)
        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                masks[index][row] = 0.0
            elif row > steps - 1 and abs(single_EOD[row - steps][-1] + 1234) \
                    > 1e-8:
                ground_truth[index][row] = \
                    (single_EOD[row][-1] - single_EOD[row - steps][-1]) / \
                    single_EOD[row - steps][-1]
            for col in range(single_EOD.shape[1]):
                if abs(single_EOD[row][col] + 1234) < 1e-8:
                    single_EOD[row][col] = 1.1
        eod_data[index, :, :] = single_EOD[:, 1:]
        base_price[index, :] = single_EOD[:, -1]
    return eod_data, masks, ground_truth, base_price


class CustomDataset(Dataset):
    def __init__(self, path, emb_file, tickers, start, end):
        self.embedding = np.load(os.path.join(path, '..', 'pretrain', emb_file))[:, start:end, :]
        self.ticker = np.genfromtxt(os.path.join(path, '..', tickers), dtype=str, delimiter='\t', skip_header=False)
        _, self.mask, self.gt, self.price_data = load_EOD_data(path, 'NASDAQ', self.ticker)
        self.mask = self.mask[:, start:end]
        self.gt = self.gt[:, start:end]
        self.price_data = self.price_data[:, start:end]
        print('mask shape = ', self.mask.shape)
        print('gt shape = ', self.gt.shape)
        print('price_data shape = ', self.price_data.shape)

    def __len__(self):
        return len(self.embedding[0])

    def __getitem__(self, idx):
        emb = self.embedding[:, idx]
        label = self.gt[:, idx]
        price = self.price_data[:, idx]
        mask = self.mask[:, idx]
        return emb, label, price, mask


def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask) ** 2 \
                         / np.sum(mask)
    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0

    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top1 = set()
        gt_top5 = set()
        gt_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top1) < 1:
                gt_top1.add(cur_rank)
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)

        rank_pre = np.argsort(prediction[:, i])

        pre_top1 = set()
        pre_top5 = set()
        pre_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top1) < 1:
                pre_top1.add(cur_rank)
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)

        # calculate mrr of top1
        top1_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top1_pos_in_gt += 1
                if cur_rank in pre_top1:
                    break
        if top1_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top += 1.0 / top1_pos_in_gt

        # back testing on top 1
        real_ret_rat_top = ground_truth[list(pre_top1)[0]][i]
        bt_long += real_ret_rat_top

        # back testing on top 5
        real_ret_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= 5
        bt_long5 += real_ret_rat_top5

        # back testing on top 10
        real_ret_rat_top10 = 0
        for pre in pre_top10:
            real_ret_rat_top10 += ground_truth[pre][i]
        real_ret_rat_top10 /= 10
        bt_long10 += real_ret_rat_top10

    performance['mrrt'] = mrr_top / (prediction.shape[1] - all_miss_days_top)
    performance['btl'] = bt_long
    # performance['btl5'] = bt_long5
    # performance['btl10'] = bt_long10
    return performance


if __name__ == '__main__':
    device = "cuda"
    net = GraphNet().to(device)
    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    desc = 'train a relational rank lstm model'
    path = 'Temporal_Relational_Stock_Ranking/training/../data/2013-01-01'
    market_name = 'NASDAQ'
    length = 16
    hidden_units = 64
    steps = 1
    lr = 0.001
    alpha = 0.1
    g_pu = 0
    emb_file = 'NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy'
    rel_name = 'wikidata'
    inner_prod = 0
    tickers = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'

    parameters = {'seq': int(length), 'unit': int(hidden_units), 'lr': float(lr),
                  'alpha': float(alpha)}
    # print('arguments:', args)
    print('parameters:', parameters)

    em = np.load(os.path.join(path, '..', 'pretrain', emb_file))
    print('embedding shape:', em.shape)
    # print(em)

    batch_size = 16

    trainset = CustomDataset(path=path, emb_file=emb_file, tickers=tickers, start=0, end=756)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = CustomDataset(path=path, emb_file=emb_file, tickers=tickers, start=756, end=1008)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    relation_encoding = np.load(
        'Temporal_Relational_Stock_Ranking/data/relation/sector_industry/NASDAQ_industry_relation.npy')

    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    # combine all the graphs to one graph
    graph_flags = np.equal(np.zeros(rel_shape, dtype=int),
                           np.sum(relation_encoding, axis=2))
    graph = np.where(graph_flags, np.zeros(rel_shape), np.ones(rel_shape))
    # calculate the edge_index
    edge_index = [[], []]
    #for i in range(len(graph)):
        #for j in range(len(graph[0])):
            #if graph[i, j] == 1:
                #edge_index[0].append(i)
                #edge_index[1].append(j)

    relation_encoding_wiki = np.load(
        'Temporal_Relational_Stock_Ranking/data/relation/wikidata/NASDAQ_wiki_relation.npy')
    rel_shape_wiki = [relation_encoding_wiki.shape[0], relation_encoding_wiki.shape[1]]
    # combine all the graphs to one graph
    graph_flags_wiki = np.equal(np.zeros(rel_shape_wiki, dtype=int),
                                np.sum(relation_encoding_wiki, axis=2))
    graph_wiki = np.where(graph_flags_wiki, np.zeros(rel_shape_wiki), np.ones(rel_shape_wiki))

    #for i in range(len(graph_wiki)):
        #for j in range(len(graph_wiki[0])):
            #if graph_wiki[i, j] == 1:
                #edge_index[0].append(i)
                #edge_index[1].append(j)
    for i in range(1026):
        for j in range(1026):
            edge_index[0].append(i)
            edge_index[1].append(j)

    edge_index = np.array(edge_index)
    print(edge_index.shape)  # (2, 53612)
    edge_index = torch.tensor(edge_index).to(device)

    for epoch in range(150):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            # data = data.to(device)
            inputs, labels, price, mask = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            price = price.to(device)
            mask = mask.to(device)
            all_one = torch.ones((inputs.shape[0], 1)).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print(inputs.get_device())
            # print(edge_index.get_device())
            # print(inputs.shape)
            outputs, output_feature = net(inputs, edge_index.to(device))
            # print(price.shape)
            return_ratio = torch.divide(torch.subtract(outputs, price), price)
            reg_loss = mse(labels, return_ratio)
            pre_pw_dif = torch.subtract(
                torch.matmul(return_ratio.T, all_one),
                torch.matmul(all_one.T, return_ratio)
            )
            gt_pw_dif = torch.subtract(
                torch.matmul(all_one.T, labels),
                torch.matmul(labels.T, all_one)
            )
            mask_pw = torch.matmul(mask.T, mask)
            a = F.relu(torch.multiply(torch.multiply(pre_pw_dif, gt_pw_dif), mask_pw))
            # print(a.numpy())
            rank_loss = np.mean(a.cpu().detach().numpy())
            loss = reg_loss + 0.1 * rank_loss
            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print(f'[{epoch + 1}] loss: {running_loss}')

    print('Finished Training')
    torch.save(net.state_dict(), 'graph_fully_weight_150_epoch.pt')

    cur_valid_pred = np.zeros([1026, 1008 - 756], dtype=float)
    cur_valid_gt = np.zeros([1026, 1008 - 756], dtype=float)
    cur_valid_mask = np.zeros([1026, 1008 - 756], dtype=float)
    val_loss = 0.0
    val_reg_loss = 0.0
    val_rank_loss = 0.0
    net.eval()
    for i, data in tqdm(enumerate(testloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        # data = data.to(device)
        inputs, labels, price, mask = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        price = price.to(device)
        mask = mask.to(device)
        all_one = torch.ones((inputs.shape[0], 1)).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        outputs, output_feature = net(inputs, edge_index)
        return_ratio = torch.divide(torch.subtract(outputs, price), price)
        val_reg_loss = mse(labels, return_ratio)
        pre_pw_dif = torch.subtract(
            torch.matmul(return_ratio.T, all_one),
            torch.matmul(all_one.T, return_ratio)
        )
        gt_pw_dif = torch.subtract(
            torch.matmul(all_one.T, labels),
            torch.matmul(labels.T, all_one)
        )
        mask_pw = torch.matmul(mask.T, mask)
        a = F.relu(torch.multiply(torch.multiply(pre_pw_dif, gt_pw_dif), mask_pw))
        # print(a.numpy())
        val_rank_loss = np.mean(a.cpu().detach().numpy())
        val_loss = reg_loss + 0.1 * rank_loss

        # print statistics
    print('val_loss: ', val_loss)
    print('val_reg_loss: ', val_reg_loss)
    print('val_rank_loss: ', val_rank_loss)

    cur_valid_pred = copy.copy(return_ratio)
    cur_valid_gt = copy.copy(labels)
    cur_valid_mask = copy.copy(mask)

    cur_valid_perf = evaluate(cur_valid_pred.cpu().detach().numpy(), cur_valid_gt.cpu().detach().numpy(),
                              cur_valid_mask.cpu().detach().numpy())
    print('Valid preformance:', cur_valid_perf)

