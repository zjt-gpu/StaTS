export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/StaTS.py \
   --dataset_type="Traffic" \
   --device="cuda:0" \
   --batch_size=4 \
   --horizon=1 \
   --layers=1 \
   --pred_len=192 \
   --windows=168 \
   --epochs=50 \
   runs --seeds='[1, 2, 3]'
