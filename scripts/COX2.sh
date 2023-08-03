python train.py --ngpu 1 \
                --dataset COX2 \
                --batch_size 256 \
                --epoch 30 \
                --dropout_rate 0.0 \
                --tatic jump \
                --embedding_dim 20

python evaluate.py --ngpu 1 \
                   --dataset COX2 \
                   --batch_size 256 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 20 \
                   --ckpt save/COX2_jump_1/save_29.pt