mkdir -p output_models
mkdir -p dsp_logs

# Indirect : Amazon to Imdb
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_language_modeling --train_data_file datasets/amazon/train.txt --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --per_gpu_train_batch_size 4 --gradient_accumulation_steps 512 --model_name_or_path roberta-base --eval_data_file datasets/imdb_data/dev.txt  --do_eval  --evaluate_during_training  --do_train --num_train_epochs 50 --learning_rate 0.0005 --logging_steps 2000 --base_task_dataset_file datasets/imdb_data/train.jsonl --dev_task_file datasets/imdb_data/dev.jsonl --test_task_file datasets/imdb_data/test.jsonl --num_samples_for_basis 32  --num_basis 5 --overwrite_output_dir --pca_every 1 --n_subspace_layers -12 --eta_set "(1.0, 1.0, -1.0)" --classifier_dropout 0.2 --classf_lr 1e-4 --lm_mt_task_weight 0.02 --output_dir output_models/roberta-imdb-tapt_1.1.-1_vwgt_0.02_indirect_amazn_to_imdb &> dsp_logs/roberta-imdb-tapt_1.1.-1_vwgt_0.02_indirect_amazn_to_imdb.txt &

# Indirect : Imdb to Amazon
CUDA_VISIBLE_DEVICES=1 python -u -m scripts.run_language_modeling --train_data_file datasets/imdb_data/all_training_joined.txt --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --per_gpu_train_batch_size 4 --gradient_accumulation_steps 512 --model_name_or_path roberta-base --eval_data_file datasets/amazon/dev.txt  --do_eval  --evaluate_during_training  --do_train --num_train_epochs 50 --learning_rate 0.0005 --logging_steps 2000 --base_task_dataset_file datasets/amazon/train.jsonl --dev_task_file datasets/amazon/dev.jsonl --test_task_file datasets/amazon/test.jsonl --num_samples_for_basis 32  --num_basis 5 --overwrite_output_dir --pca_every 1 --n_subspace_layers -12 --eta_set "(1.0, 1.0, -1.0)" --classifier_dropout 0.2 --classf_lr 1e-4 --lm_mt_task_weight 0.1 --output_dir output_models/roberta-amazon-tapt_1.1.-1_vwgt_0.1_indirect_imdb_to_amazn &> dsp_logs/roberta-amazon-tapt_1.1.-1_vwgt_0.1_indirect_imdb_to_amazn.txt &