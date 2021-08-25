import click
import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.build_features import build_features

def train_model(dataset, model, project_dir):
    """
    Train `model` on `dataset` available under `project_dir`.
    """
    train_map = {
        'grail_qa': {
            'lr': train_lr_on_grail_qa
        }
    }
    train_map[dataset][model](project_dir)

def train_lr_on_grail_qa(project_dir):
    """
    Train a LogisticRegression on grail_qa and save it under the top-level models/ dir.
    """
    df = pd.read_csv(project_dir/'data/processed/grail_qa_train.csv')
    x, y = build_features(df)

    clf = LogisticRegression()
    clf.fit(x, y)

    with open(project_dir/'models/grail_qa_lr.pkl', 'wb') as out:
        pickle.dump(clf, out)

@click.command()
@click.argument('dataset')
@click.argument('model')
def main(dataset, model):
    if dataset not in ['grail_qa']:
        raise ValueError(f'{dataset} not available')
    if model not in ['lr']:
        raise ValueError(f'{model} not available')

    logger = logging.getLogger(__name__)
    logger.info(f'Training {model} on {dataset}')

    project_dir = Path(__file__).resolve().parents[2]
    train_model('grail_qa', 'lr', project_dir)

    logger.info(f'Training complete. Model is available under the top-level models/ dir.')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
