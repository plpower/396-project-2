import csv
import numpy as np

## --------------- NOTES ------------------ ##
# The idea here is to compare the regrest of the two algorithms
# Must implement the algorithms on random data
# Calculate thier regrets (REGRET = 1/n * [OPT - ALG])
# Tune learning rates & compare


def exponential_weights(test_data):
    action1 = test_data[1]
    action2 = test_data[2]

    # regret is 2 * h sqrt(ln(k) / h)
    # learning rate is sqrt(ln(k) / h)

    # for idx in range(len(action1)):
    return None


def follow_perturbed_leader(test_data):
    # get values for each action at every round
    action1 = test_data[1]
    action2 = test_data[2]

    # add a halicination at round 0
    action1.insert(0, 3)
    action2.insert(0, 2)

    # loop through and at each day choose the payoff of the BIH action
    # key here is that we take into account the round 0 hallucinations
    ftpl = 0
    for idx in range(len(action1)):
        bih1, bih2 = best_in_hindsight(action1, action2, idx)
        if bih2 > bih1:
            ftpl += action2[idx]
        else:
            ftpl += action1[idx]
    
    # now we calculated payoff for FTPL
    # can further compare with OPT to get regret

def best_in_hindsight(action1, action2, curr_round):
    # best in hindsight DOES NOT include the current round
    # to find BIH of entire action, all it on curr_round = length of list

    bih1 =  sum(action1[:curr_round])
    bih2 = sum(action2[:curr_round])

    return bih1, bih2


if __name__ == "__main__":
    test_data = {
        1: [0.5, 1, 0, 0, 1],
        2: [0, 1, 1, 1, 1]
    }

    # implement the algorithms & calculate rergret
    exponential_weights(test_data)
    follow_perturbed_leader(test_data)

    # compare the regrets of these two alorithms againist each other
