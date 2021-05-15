
taskid=$1
metric=$2
echo "Evaluating performance on "$taskid
datasz=(10)
for k in "${datasz[@]}"
do
	echo 'Evaluating DAPT for sz = '$datasz
	mkdir -p eval_logs/small_dapt_from_scratch/$taskid
	modelfldr='/home/ec2-user/internship/dsp/m4m_dsp/'$taskid'/small_dapt_from_scratch/'$k'xTapt/'
	mv $modelfldr/dapt_$k $modelfldr/roberta-dapt_$k 
	python -m scripts.train --config training_config/classifier.jsonnet  --serialization_dir  model_logs/'DAPT_'$taskid'_'$k'xTAPT' --hyperparameters ROBERTA_CLASSIFIER_SMALL  --dataset $taskid --model $modelfldr/roberta-dapt_$k --gpu_id 0  --perf $metric --evaluate_on_test &>  eval_logs/small_dapt_from_scratch/$taskid/dapt_$k.txt

	echo 'Evaluating DAPT+TAPT for sz = '$datasz
	python -m scripts.train --config training_config/classifier.jsonnet  --serialization_dir  model_logs/'DAPT-TAPT_'$taskid'_'$k'xTAPT' --hyperparameters ROBERTA_CLASSIFIER_SMALL  --dataset $taskid --model $modelfldr/roberta-dapt-tapt_$k --gpu_id 0  --perf $metric --evaluate_on_test &> eval_logs/small_dapt_from_scratch/$taskid/roberta-dapt-tapt_$k.txt
done