#!/bin/bash
python ./scripts/precompute_video_keyframes.py --video_dir ./datasets/MSR-VTT/video --output_dir ./data/msrvtt/clip_keyframes --pattern "*.mp4"
python ./scripts/train_video_joint.py --config ./config/joint_train.yaml --device cuda:1