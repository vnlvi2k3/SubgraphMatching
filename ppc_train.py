import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import utils
from collections import defaultdict
from ppc_dataset import BaseDataset, collate_fn, UnderSampler
from kabsch import kabsch_rmsd
from ppc_model import gnn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

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


#new args:
parser.add_argument('-nonlin', type=str, default='lkyrelu', choices=['swish', 'lkyrelu'])
parser.add_argument('-cross_msgs', default=True, action='store_true')
parser.add_argument('-layer_norm', type=str, default='LN', choices=['0', 'BN', 'LN'])
parser.add_argument('-layer_norm_coors', type=str, default='0', choices=['0', 'LN'])
parser.add_argument('-final_h_layer_norm', type=str, default='0', choices=['0', 'GN', 'BN', 'LN'])
parser.add_argument('-use_dist_in_layers', default=True, action='store_true')
parser.add_argument('-skip_weight_h', type=float, default=0.5, required=False)
parser.add_argument('-leakyrelu_neg_slope', type=float, default=1e-2, required=False)
parser.add_argument('-x_connection_init', type=float, default=0., required=False)

parser.add_argument('-noise_decay_rate', type=float, default=0., required=False)
parser.add_argument('-noise_initial', type=float, default=0., required=False)
parser.add_argument('-residue_emb_dim', type=int, default=64, required=False, help='embedding')
parser.add_argument('-iegmn_lay_hid_dim', type=int, default=64, required=False)
parser.add_argument('-num_att_heads', type=int, default=50, required=False)
parser.add_argument('-iegmn_n_lays', type=int, default=5, required=False)


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
        collate_fn=partial(collate_fn),
    )
    test_dataloader = DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn),
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

            graph, cross_graph = sample
            print("batch num nodes:\n", graph.batch_num_nodes())


            # Train neural network
            # pred, attn_loss, rmsd_loss, pairdst_loss = model(
            #     X=(graph, cross_graph, V), attn_masking=(M, S), training=True
            # )
                        
            # loss = loss_fn(pred, Y) + attn_loss + rmsd_loss, pairdst_loss
            loss = 0
            loss.backward()
            optimizer.step()

            # Collect loss, true label and predicted label
            train_losses.append(loss.data.cpu().item())
            # train_true.append(Y.data.cpu().numpy())
            # train_pred.append(pred.data.cpu().numpy())

        model.eval()
        st_eval = time.time()

        for sample in tqdm(test_dataloader):

            graph, cross_graph = sample

            # Train neural network
            # pred, attn_loss, rmsd_loss, pairdst_loss = model(
            #     X=(graph, cross_graph, V), attn_masking=(M, S), training=True
            # )
                        
            # loss = loss_fn(pred, Y) + attn_loss + rmsd_loss, pairdst_loss
            loss = 0
            loss.backward()
            optimizer.step()

            # Collect loss, true label and predicted label
            test_losses.append(loss.data.cpu().item())
            # test_true.append(Y.data.cpu().numpy())
            # test_pred.append(pred.data.cpu().numpy())


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
