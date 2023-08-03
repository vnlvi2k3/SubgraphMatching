python train.py --ngpu 1 \
                --dataset DHFR \
                --batch_size 256 \
                --epoch 30 \
                --dropout_rate 0.0 \
                --tatic jump \
                --embedding_dim 71

python evaluate.py --ngpu 1 \
                   --dataset DHFR \
                   --batch_size 256 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 71 \
                   --ckpt save/DHFR_jump_1/save_29.pt