# 10X Faster Subgraph Matching: Dual Matching Networks with Interleaved Diffusion Attention

This repository contains the source code and datasets for our paper:
"10X Faster Subgraph Matching: Dual Matching Networks with Interleaved Diffusion Attention". which is accepted by IJCNN 2023.

# Requirements
We use conda to manage the environment. Please install the dependencies by running:
```
conda env create -f environment.yml
```

# Prepare the datasets
We experiment on both synthetic and real-world datasets. Before running the experiments, please follow the below steps to prepare the datasets.

## Synthetic datasets
First, we need to make configurations for a dataset and put it in the [configs](data_synthesis/configs) folder. For example, we can create a configuration file for the `synth_1` dataset as follows:
```
{
    "number_source": 5000,
    "number_subgraph_per_source": 2000,
    "avg_source_size": 30,
    "std_source_size": 5,
    "avg_degree": 3.5,
    "std_degree": 0.5,
    "number_label_node": 20,
    "number_label_edge": 10
}
```
Then, we can generate the dataset by running:
```
cd data_synthesis
python generate_dataset.py --config configs/synth_1.json
cd ..
python process_data.py synth_1 synthesis
```

## Real-world datasets
We have prepared the real-world datasets in the [data_real](data_real/datasets) folder. To generate the datasets, please run:
```
cd data_real
python make_datasets.py --ds [DATASET_NAME]
python generate_data_v1.py --config configs/[DATASET_NAME].json
cd ..
python process_data.py [DATASET_NAME] real
```

# Run the experiments
Here is the command to run the experiments:
```
python train.py [-h] [--lr LR] 
                [--epoch EPOCH] 
                [--ngpu NGPU] 
                [--dataset DATASET] 
                [--batch_size BATCH_SIZE] 
                [--num_workers NUM_WORKERS] 
                [--embedding_dim EMBEDDING_DIM] 
                [--tatic {static,cont,jump}]
                [--nhop NHOP] 
                [--n_graph_layer N_GRAPH_LAYER] 
                [--d_graph_layer D_GRAPH_LAYER] 
                [--n_FC_layer N_FC_LAYER] 
                [--d_FC_layer D_FC_LAYER] 
                [--data_path DATA_PATH] 
                [--save_dir SAVE_DIR]
                [--log_dir LOG_DIR] 
                [--dropout_rate DROPOUT_RATE] 
                [--al_scale AL_SCALE] 
                [--ckpt CKPT] 
                [--train_keys TRAIN_KEYS] 
                [--test_keys TEST_KEYS]

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --epoch EPOCH         epoch
  --ngpu NGPU           number of gpu
  --dataset DATASET     dataset
  --batch_size BATCH_SIZE
                        batch_size
  --num_workers NUM_WORKERS
                        number of workers
  --embedding_dim EMBEDDING_DIM
                        node embedding dim aka number of distinct node label
  --tatic {static,cont,jump}
                        tactic of defining number of hops
  --nhop NHOP           number of hops
  --n_graph_layer N_GRAPH_LAYER
                        number of GNN layer
  --d_graph_layer D_GRAPH_LAYER
                        dimension of GNN layer
  --n_FC_layer N_FC_LAYER
                        number of FC layer
  --d_FC_layer D_FC_LAYER
                        dimension of FC layer
  --data_path DATA_PATH
                        path to the data
  --save_dir SAVE_DIR   save directory of model parameter
  --log_dir LOG_DIR     logging directory
  --dropout_rate DROPOUT_RATE
                        dropout_rate
  --al_scale AL_SCALE   attn_loss scale
  --ckpt CKPT           Load ckpt file
  --train_keys TRAIN_KEYS
                        train keys
  --test_keys TEST_KEYS
                        test keys
```

Additionally, we have prepared the scripts to run the experiments using real datasets in the [scripts](scripts) folder. To run the experiments, please execute the following command:
```
bash scripts/[DATASET_NAME].sh
```

# Citation
If you find this repository useful in your research, please cite our paper:
```
@inproceedings{Nguyen2023xDualSM,
  author={Nguyen, Thanh Toan and Nguyen, Quang Duc and Ren, Zhao and Jo, Jun and Nguyen, Quoc Viet Hung and Nguyen, Thanh Tam},
  booktitle={2023 International Joint Conference on Neural Networks}, 
  title={10X Faster Subgraph Matching: Dual Matching Networks with Interleaved Diffusion Attention}, 
  year={2023},
  volume={},
  number={},
  pages={},
  doi={}
}
```