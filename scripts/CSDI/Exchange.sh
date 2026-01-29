export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/CSDI.py \
   --dataset_type="ExchangeRate" \
   --device="cuda:4" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   --epochs=50 \
   --warm_epochs=5 \
   runs --seeds='[1, 2, 3]'
