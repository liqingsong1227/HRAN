python run.py -epoch 1500 -name HRAN_ConvD -model hran\
      -hid_drop 0.1 -gcn_drop 0.3 \
      -score_func convd -data WN18RR  

if [ $? -ne 0 ]; then
      echo "hran_convd failed"
      exit 1
else
      echo "hran_convd succed"
fi

python run.py -epoch 1500 -name HRAN_TransE -model hran\
      -hid_drop 0.1 -gcn_drop 0.3 \
      -score_func transe -data WN18RR  \

if [ $? -ne 0 ]; then
      echo "hran_transe failed"
      exit 1
else
      echo "hran_transe succed"
fi 

python run.py -epoch 1500 -name HRAN_ConvE -model hran\
      -hid_drop 0.1 -gcn_drop 0.3 \
      -score_func conve -data WN18RR  \

if [ $? -ne 0 ]; then
      echo "hran_conve failed"
      exit 1
else
      echo "hran_conve succed"
fi

python run.py -epoch 1500 -name HRAN_Conv_Transe -model hran\
      -hid_drop 0.1 -gcn_drop 0.3 \
      -score_func conv_transe -data WN18RR  \

if [ $? -ne 0 ]; then
      echo "hran_conv_transe failed"
      exit 1
else
      echo "hran_conv_transe succed"
fi

python run.py -epoch 1500 -name DistMult -model hran\
      -hid_drop 0.1 -gcn_drop 0.3 \
      -score_func distmult -data WN18RR  \

if [ $? -ne 0 ]; then
      echo "hran_distmult failed"
      exit 1
else
      echo "hran_distmult succed"
fi

exit 0