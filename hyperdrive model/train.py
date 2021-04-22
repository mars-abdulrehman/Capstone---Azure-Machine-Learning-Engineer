from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
# https://docs.google.com/spreadsheets/d/1nefE1jns3Ybj1SC0NjRsAcXGJJKVxJZ6-sfCR0G5t6s/gviz/tq?tqx=out:csv&sheet=loan_data

path="https://docs.google.com/spreadsheets/d/1nefE1jns3Ybj1SC0NjRsAcXGJJKVxJZ6-sfCR0G5t6s/gviz/tq?tqx=out:csv&sheet=loan_data"
ds = TabularDatasetFactory.from_delimited_files(path = path)


def clean_data(data):
    # Dict for cleaning data

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    ownership = pd.get_dummies(x_df.home_ownership, prefix="owner")
    x_df.drop("home_ownership", inplace=True, axis=1)
    x_df = x_df.join(ownership)
    
    purpose1 = pd.get_dummies(x_df.purpose, prefix="purpose")
    x_df.drop("purpose", inplace=True, axis=1)
    x_df = x_df.join(purpose1)

    y_df = x_df.pop("bad_loans")
    
    return x_df, y_df


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="The number of trees in the forest.")
    parser.add_argument('--max_depth', type=int, default=None, help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")

    args = parser.parse_args()

    run.log("Number of estimators:", np.float(args.n_estimators))
    run.log("Max depth:", np.int(args.max_depth))

    model = RandomForestClassifier (n_estimators=args.n_estimators, max_depth=args.max_depth).fit(x_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:,1], average="weighted")
    run.log("AUC_weighted", np.float(auc))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')
    

x, y = clean_data(ds)

### YOUR CODE HERE ###a
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

run = Run.get_context()

if __name__ == '__main__':
    main()
    
    