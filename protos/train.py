import pandas  as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from load_data import load_train_data, load_test_data

SAMPLE_SUBMIT_FILE='../input/sample_submission.csv'
DIR='result_tmp'

if __name__=='__main__':
    df=load_train_data()

    x_train=df.drop('target', axis=1)
    y_train=df['target'].values

    use_cols=x_train.columns.values

    clf = LogisticRegression(random_state=0)
    clf.fit(x_train, y_train)

    df = load_test_data()

    x_test=df[use_cols].sort_values('id')
    pred_test=clf.predict_proba(df)

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['target']=pred_test

    df_submit.to_csv(DIR + 'submit.csv', index=False)


