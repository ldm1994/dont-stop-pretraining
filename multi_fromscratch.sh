

task=$1
metric=$2
gpuid=$3
dp=$4

CUDA_VISIBLE_DEVICES=$gpuid python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/$task/domain.10xTAPT.txt --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --from-scratch --model_name_or_path roberta-base --eval_data_file datasets/$task/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0005 --block_size 512 --logging_steps 5000 --dev_task_file datasets/$task/dev.jsonl --test_task_file datasets/$task/test.jsonl --classifier_dropout $dp --classf_lr 1e-5 --primary_task_id $task --alpha_update_algo default  --classf_ft_lr 5e-6 --classf_max_seq_len 512 --classf-metric $metric --lazy-dataset --output_dir m4m_dsp/$task/small_dapt/10xTapt/default --overwrite_output_dir --seed 0 --classf_patience 10 --num_train_epochs 100 --classf_iter_batchsz 6 --per_gpu_train_batch_size 6 --gradient_accumulation_steps 42 --eval_every 20 &> static_runlogs/$task/small_dapt_from_scratch/10xTapt/default.$task'.txt'
