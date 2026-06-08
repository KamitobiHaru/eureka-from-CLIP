#!/data1/zsy/miniconda3/bin/zsh
python scripts/precompute_motion_sequences.py --config config/default3_temporal.yaml --device cuda:0 --workers 4

python '/data1/zsy/eureka-from-CLIP/scripts/train_temporal.py' --config '/data1/zsy/eureka-from-CLIP/config/default3_temporal.yaml' --bert '/data1/zsy/weights/r8_weight0.75_uniformity2/bert_epoch28_t2i59.9.pt'