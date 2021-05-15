IMDB = {
	'imdb': '/home/ec2-user/internship/dsp/datasets/imdb_data/train.jsonl'
}

IMDB_SMALL = {
	'imdb_small': '/home/ec2-user/internship/dsp/datasets/imdb_data/train.small.jsonl'
}

AMAZON = {
	'amazon': '/home/ec2-user/internship/dsp/datasets/amazon/train.jsonl'
}

CITATION = {
	'citation_intent': '/home/ec2-user/internship/dsp/datasets/citation_intent/train.jsonl'
}

CHEMPROT = {
	'chemprot': '/home/ec2-user/internship/dsp/datasets/chemprot/train.jsonl'
}

SCIIE = {
	'sciie': '/home/ec2-user/internship/dsp/datasets/sciie/train.jsonl'
}


def get_auxtask_files(task_name):
	if task_name == 'imdb':
		return IMDB
	elif task_name == 'imdb_small':
		return IMDB_SMALL
	elif task_name == 'amazon':
		return AMAZON
	elif task_name == 'citation_intent':
		return CITATION
	elif task_name == 'chemprot':
		return CHEMPROT
	elif task_name == 'sciie':
		return SCIIE
	else:
		raise ValueError