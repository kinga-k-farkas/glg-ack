import json

def get_domains_and_questions(split, dataset:str):
    ds_map = {
        'grail_qa': get_grail_qa
    }
    return ds_map[dataset](split)

def get_grail_qa(split) -> dict:
    if split not in ('train', 'dev'):
        raise(ValueError("`split` must be one of 'train' or 'dev'"))

    path = '../data/raw/grail_qa/'
    path_prefix = path + 'grailqa_v1.0_'
    with open(path_prefix + f'{split}.json', 'rb') as f:
        data = json.load(f)

    domains, questions = [], []
    for entry in data:
        domains.append(';'.join(entry['domains']))  # entry['domains'] is an array
        questions.append(entry['question'])

    return {'domains': domains, 'questions': questions}