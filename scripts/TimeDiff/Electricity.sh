export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/TimeDiff.py \
   --dataset_type="Electricity" \
   --device="cuda:7" \
   --batch_size=8 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   --epochs=50 \
   --patience=5 \
   runs --seeds='[1, 2, 3]'

