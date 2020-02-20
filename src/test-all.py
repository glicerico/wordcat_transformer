# Code taken from AdaGram WSD evaluator: https://github.com/sbos/adagram.jl
# Slightly adapted for our purposes

import numpy as np
import sys
import os
import subprocess
import itertools
from sklearn.metrics import adjusted_rand_score, v_measure_score


def get_pairs(labels):
    result = []
    for label in np.unique(labels):
        ulabels = np.where(labels == label)[0]
        for p in itertools.combinations(ulabels, 2):
            result.append(p)
    return result


def compute_fscore(true, pred):
    true_pairs = get_pairs(true)
    pred_pairs = get_pairs(pred)
    int_size = len(set(true_pairs).intersection(pred_pairs))
    # Handle special cases
    eps = 0.00001
    len_true = len(true_pairs) + eps  # Add epsilon to avoid division by 0
    len_pred = len(pred_pairs) + eps
    if len_true + len_pred == 0:  # Neither true nor pred have any pairs
        return 1
    elif int_size == 0:  # No match between pairs
        return 0
    else:
        p = int_size / float(len(pred_pairs))
        r = int_size / float(len(true_pairs))
        return 2 * p * r / float(p + r)  # Added epsilon to avoid division by 0


def read_answers(filename, all_in_one=False):
    with open(filename, 'r') as f:
        keys = []
        instances = []
        senses = []
        senses_id = {}
        sense_count = 0
        for line in f.readlines():
            key, instance, sense = line.strip().split(' ')
            if all_in_one:  # Baseline where all words have same sense
                sense = 0
            num = int(instance.split('.')[-1])
            keys.append(key)
            instances.append(num)
            senses.append(sense)
            if sense not in senses_id:
                senses_id[sense] = sense_count
                sense_count += 1
        answers = {}
        for k, i, s in zip(keys, instances, senses):
            if k not in answers:
                answers[k] = ([], [])
            answers[k][0].append(i)
            answers[k][1].append(senses_id[s])
        return answers


def compute_metrics(answers, predictions):
    aris = []
    vscores = []
    fscores = []
    weights = []
    for k, v in answers.items():
        if len(v[0]) == 1:
            print(f"Skipping one-instance word {k}")
            continue
        print(f"Evaluating word {k}")
        idx = np.argsort(np.array(v[0]))
        true = np.array(v[1])[idx]
        pred = np.array(predictions[k][1])
        weights.append(pred.shape[0])
        if len(np.unique(true)) > 1:
            aris.append(adjusted_rand_score(true, pred))
        vscores.append(v_measure_score(true, pred))
        fscores.append(compute_fscore(true, pred))
    #        print '%s: ari=%f, vscore=%f, fscore=%f' % (k, aris[-1], vscores[-1], fscores[-1])
    aris = np.array(aris)
    vscores = np.array(vscores)
    fscores = np.array(fscores)
    weights = np.array(weights)
    print('number of one-sense words: %d' % (len(vscores) - len(aris)))
    print('mean ari: %f' % np.mean(aris))
    print('mean vscore: %f' % np.mean(vscores))
    print('weighted vscore: %f' % np.sum(vscores * (weights / float(np.sum(weights)))))
    print('mean fscore: %f' % np.mean(fscores))
    print('weighted fscore: %f' % np.sum(fscores * (weights / float(np.sum(weights)))))
    return np.mean(aris), np.mean(vscores)


if __name__ == '__main__':
    GS_file = sys.argv[1]
    pred_file = sys.argv[2]
    baseline = ''
    if len(sys.argv) == 4:
        baseline = sys.argv[3]

    print("Loading gold standard...")
    true_answers = read_answers(GS_file)
    print("Loading prediction...")
    if baseline == 'all_in_one':
        print("entered")
        predictions = read_answers(pred_file, all_in_one=True)
    else:
        predictions = read_answers(pred_file)

    print("Evaluating...")
    compute_metrics(true_answers, predictions)
