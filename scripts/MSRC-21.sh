python train.py --ngpu 1 \
                --dataset MSRC-21 \
                --batch_size 64 \
                --epoch 30 \
                --dropout_rate 0.0 \
                --tatic jump \
                --embedding_dim 141

python evaluate.py --ngpu 1 \
                   --dataset MSRC-21 \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 141 -\
                   -ckpt save/MSRC-21_jump_1/save_20.pt