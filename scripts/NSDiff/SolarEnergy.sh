export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/NsDiff.py \
   --dataset_type="SolarEnergy" \
   --device="cuda:2" \
   --batch_size=4 \
   --horizon=1 \
   --layers=1 \
   --pred_len=192 \
   --windows=168 \
   --rolling_length=24 \
   --load_pretrain=False \
   --epochs=50 \
   --patience=5 \
   runs --seeds='[1, 2, 3]'

