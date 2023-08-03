python generate_data_v1.py --config configs/large.json
python process_data.py large
python train.py --ngpu 1 --dataset large --batch_size 256 --epoch 30 --dropout_rate 0.0 