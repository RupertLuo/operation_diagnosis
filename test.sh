#!/bin/zsh
# echo "== Graph ============ Context[ False ] ========= Procedure KG[ False ] ====== main_feature[False] ====="
# python code/model_code_v2/test.py -p 0 -c 0 -main_feature 0 --batch_size 12
echo "== Graph ============ Context[ False ] ========= Procedure KG[ False ] ====== main_feature[False] ===== TransformerNet"
python code/model_code_v2/test.py -p 0 -c 0 -main_feature 1 --batch_size 12 --network TransformerNet
echo "== Graph ============ Context[ False ] ========= Procedure KG[ True ] ====== main_feature[False] ===== TransformerNet"
python code/model_code_v2/test.py -p 1 -c 0 -main_feature 1 --batch_size 12 --network TransformerNet
echo "== Graph ============ Context[ True ] ========= Procedure KG[ False ] ====== main_feature[False] ===== TransformerNet"
python code/model_code_v2/test.py -p 0 -c 1 -main_feature 1 --batch_size 12 --network TransformerNet
echo "== Graph ============ Context[ True ] ========= Procedure KG[ True ] ====== main_feature[False] ===== TransformerNet"
python code/model_code_v2/test.py -p 1 -c 1 -main_feature 1 --batch_size 12 --network TransformerNet