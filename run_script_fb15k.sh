nohup python run.py -epoch 1500 -name HRAN_ConvD -model hran \
      -hid_drop 0 -gcn_drop 0 \
      -score_func convd -data FB15k-237 -batch 256 -lr 0.00025 \
      -test_batch 256 

if [ $? -ne 0 ]; then
      echo "hran_convd failed"
      exit 1
else
      echo "hran_convd succed"
fi

nohup python run.py -epoch 1500 -name HRAN_TransE -model hran \
      -hid_drop 0 -gcn_drop 0 \
      -score_func transe -data FB15k-237 -batch 256 -lr 0.00025 \
      -test_batch 256

if [ $? -ne 0 ]; then
      echo "hran_transe failed"
      exit 1
else
      echo "hran_transe succed"
fi 

nohup python run.py -epoch 1500 -name HRAN_ConvE -model hran\
      -hid_drop 0 -gcn_drop 0 \
      -score_func conve -data FB15k-237  -batch 256 -lr 0.00025 \
      -test_batch 256 

if [ $? -ne 0 ]; then
      echo "hran_conve failed"
      exit 1
else
      echo "hran_conve succed"
fi

nohup python run.py -epoch 1500 -name HRAN_Conv_Transe -model hran\
      -hid_drop 0 -gcn_drop 0 \
      -score_func conv_transe -data FB15k-237 -batch 256 -lr 0.00025 \
      -test_batch 256 

if [ $? -ne 0 ]; then
      echo "hran_conv_transe failed"
      exit 1
else
      echo "hran_conv_transe succed"
fi

nohup python run.py -epoch 1500 -name DistMult -model hran \
      -hid_drop 0 -gcn_drop 0 \
      -score_func distmult -data FB15k-237  -batch 256 -lr 0.00025 \
      -test_batch 256 

if [ $? -ne 0 ]; then
      echo "hran_distmult failed"
      exit 1
else
      echo "hran_distmult succed"
fi

exit 0