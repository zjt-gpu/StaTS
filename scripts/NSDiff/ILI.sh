export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/NsDiff.py \
   --dataset_type="ILI" \
   --device="cuda:2" \
   --batch_size=16 \
   --horizon=1 \
   --num_preprocess_cells=1 \
   --pred_len=36 \
   --windows=168 \
   --epochs=50 \
   --patience=5 \
   runs --seeds='[1, 2, 3]'