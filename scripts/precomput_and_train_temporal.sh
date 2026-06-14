#!/data1/zsy/miniconda3/bin/zsh
python /data1/zsy/eureka-from-CLIP/scripts/precompute_video_keyframes.py --video_dir /data2/zsy/datasets/MSR-VTT/video --output_dir /data1/zsy/eureka-from-CLIP/data/msrvtt/clip_keyframes --pattern "*.mp4"
python /data1/zsy/eureka-from-CLIP/scripts/train_video_joint.py --config /data1/zsy/eureka-from-CLIP/config/joint_train.yaml --device cuda:1