#!/bin/zsh
# echo "== TRM ============ Context[ False ] ========= Procedure KG[ True ]====== standard_trm [False] ===== TransformerNet" >> log/result.log
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 code/model_code_v2/train.py \
# -mul_gpu 1 \
# -p 0 \
# -c 0 \
# -main_feature 1 \
# --epoch_size 100 \
# --batch_size 12 \
# --network TransformerNet \
# -task 2 \
# -load_dataset 1 \
# -log log/task2_trm_0_0_result.log

echo "== TRM ============ Context[ False ] ========= Procedure KG[ True ]====== standard_trm [False] ===== TransformerNet" >> log/result.log
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 code/model_code/train.py \
-mul_gpu 1 \
-p 0 \
-c 0 \
-main_feature 1 \
--epoch_size 50 \
--batch_size 8 \
--network TransformerNet \
--standard_trm 1 \
-task 2 \
-load_dataset 1 \
-log log/task2_trm_stand_result.log

echo "== TRM ============ Context[ False ] ========= Procedure KG[ True ]====== standard_trm [False] ===== TransformerNet" >> log/result.log
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 code/model_code/train.py \
-mul_gpu 1 \
-p 0 \
-c 0 \
-main_feature 1 \
--epoch_size 50 \
--batch_size 8 \
--network TransformerNet \
--standard_trm 1 \
-task 2 \
-load_dataset 1 \
-log log/task2_trm_pos_result.log






