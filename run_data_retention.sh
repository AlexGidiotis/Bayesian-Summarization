python data_retention.py \
    --data xsum cnn_dailymail --root_path exp_runs \
    --models PEGASUS PEGASUS BART BART --n_list 10 20 10 20 \
    --bases PEGASUS BART