#/usr/bin/bash
python scripts/train_bert.py --config config/ablation/ablation_mlp.yaml
python '/home/occccce/eureka-from-CLIP/scripts/train_bert_domain.py' --config '/home/occccce/eureka-from-CLIP/config/bert_domain copy.yaml'