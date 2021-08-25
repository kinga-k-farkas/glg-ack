import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def transform_labels(labels: np.ndarray):
    """
    Prepare labels array for downstream linear model.
    """
    labels[np.where(labels == 'healthcare')] = 0.
    labels[np.where(labels == 'technology')] = 1.
    return labels.astype(np.float64)

def build_features(df: pd.DataFrame) -> Tuple[np.ndarray]:
    """
    Build LogisticRegression features for grail_qa.
    """
    stop_words = ['the', 'what', 'of', 'is', 'which', 'has', 'by', 'that', 'in', 'and', 'with', 'for', 'was', 'name', 'to', 'are', 'how', 'who', 'as', 'on', 'many', 'than', 'used', 'have', 'does', 'an']
    
    tfidf = TfidfVectorizer(stop_words=stop_words)

    x = tfidf.fit_transform(df.questions)
    y = transform_labels(df.domains.values)

    # Save tfidf to transform inputs ad-hoc
    project_dir = Path(__file__).resolve().parents[2]
    with open(project_dir/'models/grail_qa_tfidf.pkl', 'wb') as out:
        pickle.dump(tfidf, out)
    
    return x, y