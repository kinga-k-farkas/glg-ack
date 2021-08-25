from typing import Iterable, Tuple
import json
import pandas as pd

def get_domains_and_questions(split: str, dataset: str, **kwargs) -> dict:
    """
    Get the `split` of the `dataset` you want, with domains and questions processed.
    """
    ds_map = {
        'grail_qa': get_grail_qa
    }
    return ds_map[dataset](split, **kwargs)

## Grail QA

def get_grail_qa(split: str, path='../data/raw/grail_qa/') -> dict:
    """
    Get domains and questions from a `split` of grail_qa.
    """
    if split not in ('train', 'dev'):
        raise(ValueError("`split` must be one of 'train' or 'dev'"))

    path_prefix = path + 'grailqa_v1.0_'
    with open(path_prefix + f'{split}.json', 'rb') as f:
        data = json.load(f)

    domains, questions = [], []
    for entry in data:
        domains.append(';'.join(entry['domains']))  # entry['domains'] is an array
        questions.append(entry['question'])

    return {'domains': domains, 'questions': questions}

def make_grail_qa(path='../data/raw/grail_qa/') -> Tuple[pd.DataFrame]:
    """
    Process the raw grail_qa dataset into the format required for this project.
    """
    from pathlib import Path

    if isinstance(path, Path):
        path = str(path) + '/'

    train = pd.DataFrame(get_domains_and_questions('train', 'grail_qa', path=path))
    dev   = pd.DataFrame(get_domains_and_questions('dev',   'grail_qa', path=path))

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

def filter_domains(data: pd.DataFrame, domains: Iterable) -> pd.DataFrame:
    parts = []
    for domain in domains:
        parts.append(data.loc[data.domains == domain])
    return pd.concat(parts)

def set_label(df: pd.DataFrame, label: str, subdomains: Iterable) -> pd.DataFrame:
    df.domains.loc[df.domains.isin(subdomains)] = label
    return df