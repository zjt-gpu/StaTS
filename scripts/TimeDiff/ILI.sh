export PYTHONPATH=./:/notebooks/pytorchtimseries
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/TimeDiff.py \
   --dataset_type="ILI" \
   --device="cuda:1" \
   --batch_size=16 \
   --horizon=1 \
   --num_preprocess_cells=1 \
   --pred_len=36 \
   --windows=168 \
   --epochs=50 \
   --patience=5 \
   runs --seeds='[1, 2, 3]'