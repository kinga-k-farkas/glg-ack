import pickle
from pathlib import Path

import streamlit as st
import pandas as pd

"""
# GLG-ACK

Our model automatically classifies client queries as being either healthcare-related or technology-related. You can test it out below.
"""

project_dir = Path(__file__).resolve().parents[2]

with open(project_dir/'models/grail_qa_tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open(project_dir/'models/grail_qa_lr.pkl', 'rb') as f:
    clf = pickle.load(f)

query = st.text_input('Enter your query:')
if len(query) == 0:
    st.write('_Prediction: _')
else:
    pred = clf.predict_proba(tfidf.transform([query]))
    pred = pd.DataFrame(pred, index=['probability'], columns=['Healthcare', 'Technology'])
    st.write(pred)
    healthcare_proba, technology_proba = pred.Healthcare[0], pred.Technology[0]
    if healthcare_proba < 0.7 or technology_proba < 0.7:
        pred = 'Other'
    else:
        pred = 'Healthcare' if healthcare_proba > technology_proba else 'Technology'
    st.write(f'_Prediction: {pred}_')
