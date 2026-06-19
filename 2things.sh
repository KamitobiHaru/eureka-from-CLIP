#!/bin/bash
python scripts/train_bert.py --config config/ablation/ablation_mlp.yaml
python ./scripts/train_bert_domain.py --config ./config/bert_domain.yaml