# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv

from src.data.utils import *

def grail_qa(project_dir):
    train, dev = make_grail_qa(path=project_dir/'data/raw/grail_qa/')

    output_dir = project_dir/'data/processed'
    output_dir.mkdir(exist_ok=True)

    train.to_csv(output_dir/'grail_qa_train.csv')
    dev.to_csv(output_dir/'grail_qa_dev.csv')

@click.command()
@click.argument('dataset')
def main(dataset):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    if dataset not in ['grail_qa']:
        raise ValueError(f'{dataset} not available')

    project_dir = Path(__file__).resolve().parents[2]

    make_map = {
        'grail_qa': grail_qa
    }

    logger = logging.getLogger(__name__)
    logger.info(f'Making {dataset} from raw data')

    make_map[dataset](project_dir)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()