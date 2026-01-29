export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/NsDiff.py \
   --dataset_type="ETTh2" \
   --device="cuda:2" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   --load_pretrain=False \
   --epochs=50 \
   --patience=5 \
   runs --seeds='[1, 2, 3]'
