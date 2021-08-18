import json
import pandas as pd

def get_domains_and_questions(split, dataset:str):
    ds_map = {
        'grail_qa': get_grail_qa
    }
    return ds_map[dataset](split)

## Grail QA

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

def make_grail_qa():
    train = pd.DataFrame(get_domains_and_questions('train', 'grail_qa'))
    dev   = pd.DataFrame(get_domains_and_questions('dev',   'grail_qa'))

    domains = ['medicine', 'computer', 'spaceflight', 'biology', 'automotive', 'internet', 'engineering']
    train = filter_domains(train, domains)
    dev   = filter_domains(dev,   domains)

    healthcare_subdomains = ['medicine', 'biology']
    technology_subdomains = ['computer', 'spaceflight', 'automotive', 'internet', 'engineering']

    train = set_label(train, 'healthcare', healthcare_subdomains)
    train = set_label(train, 'technology', technology_subdomains)
    dev   = set_label(dev,   'healthcare', healthcare_subdomains)
    dev   = set_label(dev,   'technology', technology_subdomains)

    return train, dev

##

def filter_domains(data, domains):
    parts = []
    for domain in domains:
        parts.append(data.loc[data.domains == domain])
    return pd.concat(parts)

def set_label(df, label, subdomains):
    df.domains.loc[df.domains.isin(subdomains)] = label
    return df