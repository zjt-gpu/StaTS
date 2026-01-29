export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/CSDI.py \
   --dataset_type="ILI" \
   --device="cuda:1" \
   --batch_size=16 \
   --horizon=1 \
   --pred_len=36 \
   --windows=168 \
   --patience=5 \
   --epochs=50 \
   --warm_epochs=5 \
   runs --seeds='[1, 2, 3]'