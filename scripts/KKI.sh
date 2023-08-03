python train.py --ngpu 1 \
                --dataset KKI \
                --batch_size 64 \
                --epoch 30 \
                --dropout_rate 0.0 \
                --tatic jump \
                --embedding_dim 190

python evaluate.py --ngpu 1 \
                   --dataset KKI \
                   --batch_size 64 \
                   --dropout_rate 0.0 \
                   --tatic jump \
                   --embedding_dim 190 \
                   --ckpt save/KKI_jump_1/save_29.pt