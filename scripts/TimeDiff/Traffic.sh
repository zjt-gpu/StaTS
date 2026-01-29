export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/TimeDiff.py \
   --dataset_type="Traffic" \
   --device="cuda:3" \
   --batch_size=4 \
   --pred_len=192 \
   --windows=168 \
   --epochs=50 \
   --patience=5 \
   runs --seeds='[1, 2, 3]'
