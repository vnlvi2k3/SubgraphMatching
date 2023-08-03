python train.py --ngpu 1 \
                --dataset DBLP-v1 \
                --batch_size 256 \
                --epoch 30 \
                --dropout_rate 0.0 \
                --tatic jump \
                --embedding_dim 39

python evaluate.py --ngpu 1 \
                   --dataset DBLP-v1 \
                   --batch_size 256 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 39 \
                   --ckpt save/DBLP-v1_jump_1/save_26.pt