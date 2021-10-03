python -u standard_summarization.py \
    --model_path google/pegasus-large --dataset_name xsum\
    --output_path pegasus_pre_test \
    --text_column document \
    --summary_column summary \
    --max_source_length 256 \
    --seed 100 \
    --test_batch_size 6 \
    --max_test_samples 10 \
    --num_beams 3