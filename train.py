import argparse
import os
import pickle
import time
import itertools

import numpy as np
import utils
from collections import defaultdict
from update_dataset import BaseDataset, collate_fn, UnderSampler
from kabsch import kabsch_rmsd
from gnn import gnn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default=20)
parser.add_argument("--ngpu", help="number of gpu", type=int, default=1)
parser.add_argument("--dataset", help="dataset", type=str, default="tiny")
parser.add_argument("--batch_size", help="batch_size", type=int, default=32)
parser.add_argument(
    "--num_workers", help="number of workers", type=int, default=os.cpu_count()
)
parser.add_argument(
    "--embedding_dim",
    help="node embedding dim aka number of distinct node label",
    type=int,
    default=20,
)
parser.add_argument(
    "--tatic",
    help="tactic of defining number of hops",
    type=str,
    default="static",
    choices=["static", "cont", "jump"],
)
parser.add_argument("--nhop", help="number of hops", type=int, default=1)
parser.add_argument("--branch", help="choosing branch",
                    type=str, default="both", choices=["both", "left", "right"])
parser.add_argument("--n_graph_layer",
                    help="number of GNN layer", type=int, default=4)
parser.add_argument(
    "--d_graph_layer", help="dimension of GNN layer", type=int, default=140
)
parser.add_argument(
    "--n_FC_layer", help="number of FC layer", type=int, default=4)
parser.add_argument(
    "--d_FC_layer", help="dimension of FC layer", type=int, default=128)
parser.add_argument(
    "--data_path", help="path to the data", type=str, default="data_processed"
)
parser.add_argument(
    "--save_dir", help="save directory of model parameter", type=str, default="save"
)
parser.add_argument("--log_dir", help="logging directory",
                    type=str, default="log")
parser.add_argument("--dropout_rate", help="dropout_rate",
                    type=float, default=0.0)
parser.add_argument("--al_scale", help="attn_loss scale",
                    type=float, default=1.0)
parser.add_argument("--ckpt", help="Load ckpt file", type=str, default="")
parser.add_argument(
    "--train_keys", help="train keys", type=str, default="train_keys.pkl"
)
parser.add_argument("--test_keys", help="test keys",
                    type=str, default="test_keys.pkl")
parser.add_argument("--tag", help="Additional tag for saving and logging folder",
                    type=str, default="")



def main(args):
    # hyper parameters

    num_epochs = args.epoch
    lr = args.lr
    data_path = os.path.join(args.data_path, args.dataset)
    args.train_keys = os.path.join(data_path, args.train_keys)
    args.test_keys = os.path.join(data_path, args.test_keys)
    save_dir = os.path.join(
        args.save_dir, "%s_%s_%d" % (args.dataset, args.tatic, args.nhop)
    )
    log_dir = os.path.join(
        args.log_dir, "%s_%s_%d" % (args.dataset, args.tatic, args.nhop)
    )
    if args.branch != "both":
        save_dir += "_" + args.branch
        log_dir += "_" + args.branch

    if args.tag != "":
        save_dir += "_" + args.tag
        log_dir += "_" + args.tag

    # Make save dir if it doesn't exist
    if not os.path.isdir(save_dir):
        os.system("mkdir " + save_dir)
    if not os.path.isdir(log_dir):
        os.system("mkdir " + log_dir)

    # Read data. data is stored in format of dictionary. Each key has information about protein-ligand complex.
    with open(args.train_keys, "rb") as fp:
        train_keys = pickle.load(fp)
    with open(args.test_keys, "rb") as fp:
        test_keys = pickle.load(fp)

    # Print simple statistics about dude data and pdbbind data
    print(f"Number of train data: {len(train_keys)}")
    print(f"Number of test data: {len(test_keys)}")

    # Initialize model
    # if args.ngpu > 0:
    #     cmd = utils.set_cuda_visible_device(args.ngpu)
    #     os.environ["CUDA_VISIBLE_DEVICES"] = cmd[:-1]

    model = gnn(args)
    print(
        "Number of parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.initialize_model(model, device, load_save_file=args.ckpt)

    # Train and test dataset
    train_dataset = BaseDataset(
        train_keys, data_path, embedding_dim=args.embedding_dim)
    test_dataset = BaseDataset(
        test_keys, data_path, embedding_dim=args.embedding_dim)

    # num_train_iso = len([0 for k in train_keys if 'iso' in k])
    # num_train_non = len([0 for k in train_keys if 'non' in k])
    # train_weights = [1/num_train_iso if 'iso' in k else 1/num_train_non for k in train_keys]
    # train_sampler = UnderSampler(train_weights, len(train_weights), replacement=True)

    train_dataloader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        # sampler = train_sampler
    )
    test_dataloader = DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss function
    loss_fn = nn.BCELoss()

    # Logging file
    log_file = open(
        os.path.join(log_dir, "%s_trace.csv" % args.dataset), "w", encoding="utf-8"
    )
    log_file.write(
        "epoch,train_losses,test_losses,train_roc,test_roc,train_time,test_time\n"
    )

    trainacc = []
    testacc = []
    fulltime = 0

    for epoch in range(num_epochs):
        print("EPOCH", epoch)
        st = time.time()
        # Collect losses of each iteration
        train_losses = []
        test_losses = []

        # Collect true label of each iteration
        train_true = []
        test_true = []

        # Collect predicted label of each iteration
        train_pred = []
        test_pred = []

        model.train()
        for sample in tqdm(train_dataloader):
            model.zero_grad()
            H, A1, A2, M, S, Y, V, _, _, _ = sample
            H, A1, A2, M, S, Y, V = (
                H.to(device),
                A1.to(device),
                A2.to(device),
                M.to(device),
                S.to(device),
                Y.to(device),
                V.to(device),
            )

            # Train neural network
            pred, attn_loss = model(
                X=(H, A1, A2, V), attn_masking=(M, S), training=True
            )
                 
            loss = loss_fn(pred, Y) + attn_loss
            loss.backward()
            optimizer.step()

            # Collect loss, true label and predicted label
            train_losses.append(loss.data.cpu().item())
            train_true.append(Y.data.cpu().numpy())
            train_pred.append(pred.data.cpu().numpy())

        model.eval()
        st_eval = time.time()
        sam = 0

        for sample in tqdm(test_dataloader):
            sam = sam + 1
            H, A1, A2, M, S, Y, V, P, Q, _ = sample
            H, A1, A2, M, S, Y, V = (
                H.to(device),
                A1.to(device),
                A2.to(device),
                M.to(device),
                S.to(device),
                Y.to(device),
                V.to(device),
            )

            # Test neural network
            pred, attn_loss = model(
                X=(H, A1, A2, V), attn_masking=(M, S), training=True
            )
            predict_mapping = model.get_refined_adjs2((H, A1, A2, V))

            
            
            for batch_idx, i in enumerate(pred):
                
                if i.item() >= 0.5:
                    n1 = len(V[batch_idx][V[batch_idx]==1])

                    test_true_mapping = M[batch_idx].unsqueeze(0).data.cpu().numpy()
                    test_pred_mapping = predict_mapping[batch_idx].unsqueeze(0).data.cpu().numpy()


                    for mapping_true, mapping_pred in zip(test_true_mapping, test_pred_mapping):
                        gt_mapping = {}
                        x_coord, y_coord = np.where(mapping_true > 0)
                        for x, y in zip(x_coord, y_coord):
                            if x < y:
                                gt_mapping[x] = [y]  # Subgraph node: Graph node

                        pred_mapping = defaultdict(lambda: {})
                        x_coord, y_coord = np.where(mapping_pred > 0)

                        for x, y in zip(x_coord, y_coord):
                            if x < y:
                                if y in pred_mapping[x]:
                                    pred_mapping[x][y] = (
                                        pred_mapping[x][y] + mapping_pred[x][y]
                                    ) / 2
                                else:
                                    pred_mapping[x][y] = mapping_pred[
                                        x, y
                                    ]  # Subgraph node: Graph node
                            else:
                                if x in pred_mapping[y]:
                                    pred_mapping[y][x] = (
                                        pred_mapping[y][x] + mapping_pred[x][y]
                                    ) / 2
                                else:
                                    pred_mapping[y][x] = mapping_pred[
                                        x, y
                                    ]  # Subgraph node: Graph node

                    sorted_predict_mapping = defaultdict(lambda: [])
                    sorted_predict_mapping.update(
                        {
                            k: [
                                y[0]
                                for y in sorted(
                                    [(n, prob) for n, prob in v.items()],
                                    key=lambda x: x[1],
                                    reverse=True,
                                )
                            ]
                            for k, v in pred_mapping.items()
                        }
                    )
                    
                    for k, v in sorted_predict_mapping.items():
                        sorted_predict_mapping[k] = [item for item in v if item >= n1]

                    is_iso = 0
                    topk = 9



                    div = 2
                    upper = int(np.ceil(n1/div))
                    for i in range(upper):
                        mapping_combinations = []

                        if i == upper - 1:
                            S = dict(list(sorted_predict_mapping.items())[i*div : n1])
                            sub_coords = P[batch_idx][i*div : n1]
                        else:
                            S = dict(list(sorted_predict_mapping.items())[i*div : (i+1)*div])
                            sub_coords = P[batch_idx][i*div : (i+1)*div]

                        if n1 > 0:
                            for i in S:
                                S[i] = S[i][:topk]
                            _ , values = zip(*S.items())
                            for row in itertools.product(*values):
                                mapping_combinations.append(row)
                            mapping_combinations = [list(i) for i in mapping_combinations]

                        for mapping_idx in mapping_combinations:
                            mapping_idx = [mapping_idx[j] - n1 for j in range(len(mapping_idx))]
                            graph_coords = Q[batch_idx][mapping_idx]

                            rmsd = kabsch_rmsd(sub_coords, graph_coords)
                            if rmsd <= 1e-3:
                                is_iso = is_iso + 1
                                break


                    if is_iso == upper:
                        pred[batch_idx] = torch.tensor([1.], device=device)
                    else:
                        pred[batch_idx] = torch.tensor([0.], device=device)
            
            loss = loss_fn(pred, Y) + attn_loss

            # Collect loss, true label and predicted label
            test_losses.append(loss.data.cpu().item())
            test_true.append(Y.data.cpu().numpy())
            test_pred.append(pred.data.cpu().numpy())


        end = time.time()

        train_losses = np.mean(np.array(train_losses))
        test_losses = np.mean(np.array(test_losses))

        train_pred = np.concatenate(train_pred, 0)
        test_pred = np.concatenate(test_pred, 0)

        train_true = np.concatenate(train_true, 0)
        test_true = np.concatenate(test_true, 0)

        train_roc = roc_auc_score(train_true, train_pred)
        test_roc = roc_auc_score(test_true, test_pred)

        trainacc.append(train_roc)
        testacc.append(test_roc)
        fulltime = fulltime + end - st

        print(
            "%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f"
            % (
                epoch,
                train_losses,
                test_losses,
                train_roc,
                test_roc,
                st_eval - st,
                end - st_eval,
            )
        )

        log_file.write(
            "%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n"
            % (
                epoch,
                train_losses,
                test_losses,
                train_roc,
                test_roc,
                st_eval - st,
                end - st_eval,
            )
        )
        log_file.flush()

        name = save_dir + "/save_" + str(epoch) + ".pt"
        torch.save(model.state_dict(), name)

    max_train_acc = np.max(np.array(trainacc))
    max_test_acc = np.max(np.array(testacc))
    mean_train_acc = np.mean(np.array(trainacc))
    mean_test_acc = np.mean(np.array(testacc))

    print("Full time:\t", fulltime, "max train acc:\t", max_train_acc, "max test acc:\t",
           max_test_acc, "mean train acc:\t", mean_train_acc, "mean test acc:\t", mean_test_acc)

    log_file.close()


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (
        now.tm_year,
        now.tm_mon,
        now.tm_mday,
        now.tm_hour,
        now.tm_min,
        now.tm_sec,
    )
    print(s)

    main(args)
