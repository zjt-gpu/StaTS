export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/CSDI.py \
   --dataset_type="Traffic" \
   --device="cuda:7" \
   --batch_size=4 \
   --horizon=1 \
   --layers=1 \
   --pred_len=192 \
   --windows=168 \
   --epochs=50 \
   --warm_epochs=5 \
   runs --seeds='[1, 2, 3]'
