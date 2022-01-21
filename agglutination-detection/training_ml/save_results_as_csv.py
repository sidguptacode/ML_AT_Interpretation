import numpy as np
import pandas as pd


def main():
    results = np.load('./results.npz')
    labels = results['labels']
    raw_preds = list(np.array(results['raw_preds']) - 1)
    preds = results['predictions']
    print(len(labels))
    print(len(raw_preds))

    df = pd.DataFrame({'raw_predictions': raw_preds, 'predictions': preds, 'labels': labels})
    # print(df['labels'].unique())
    # print(df['predictions'].unique())
    # assert False

    grp = df.groupby('labels')

    labels = {}
    for g, dg in grp:
        correct = dg.where(dg.predictions == dg.labels).dropna().reset_index(drop=True).raw_predictions
        incorrect = dg.where(dg.predictions != dg.labels).dropna().reset_index(drop=True).raw_predictions
        dg = pd.concat([correct, incorrect], keys=['correct', 'incorrect'], axis=1)
        labels[g] = dg

    formated = pd.concat(labels, axis=1)
    formated.to_excel('results_covid_for_swarm.xlsx')

main()