import csv
import numpy as np
from scipy.stats import geom
import math

## --------------- NOTES ------------------ ##
# The idea here is to compare the regret of the two algorithms
# Must implement the algorithms on random data
# Calculate thier regrets (REGRET = 1/n * [OPT - ALG])
# Tune learning rates & compare


def exponential_weights(test_data, epsilon):
    action1 = test_data[1]
    action2 = test_data[2]
    total_payoff = 0

    for r in range(1, len(test_data[1])+1):
        # calculate the probability of chosing every action in round r 
        probabilities, payoffs = get_probabilities(r, epsilon, test_data)
        # chose action for round r with probabilities
        action_payoff = np.random.choice(payoffs, p=probabilities)
        total_payoff += action_payoff

    # regret is 2 * h sqrt(ln(k) / h)
    # learning rate is sqrt(ln(k) / h)

    return total_payoff

def get_probabilities(r, e, test_data):
    payoffs = []
    total_payoff = 0
    probabilities = []
    for action in range(len(test_data)):
        action_payoff = sum(test_data[action + 1][:r])
        payoffs.append(action_payoff)
        print(test_data[action+1][:r])
        print(action_payoff)
        print(payoffs)
    total_payoff = sum(payoffs)

    for action in range(len(test_data)):
        probabilities.append(payoffs[action]/total_payoff)
    
    return probabilities, payoffs


def follow_perturbed_leader(test_data):
    # get values for each action at every round
    action1 = test_data[1]
    action2 = test_data[2]

    # generate hallucinations 
    hallucinations = geom.rvs(p, size=len(test_data))

    # add a halicination at round 0
    action1.insert(0, hallucinations[0])
    action2.insert(0, hallucinations[1])

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

    return ftpl

def best_in_hindsight(action1, action2, curr_round):
    # best in hindsight DOES NOT include the current round
    # to find BIH of entire action, all it on curr_round = length of list

    bih1 =  sum(action1[:curr_round])
    bih2 = sum(action2[:curr_round])

    return bih1, bih2

def theo_opt_epsilon(test_data):
    k = len(test_data)
    n = len(test_data[1])
    epsilon = math.sqrt(np.log(k)/n)
    
    return epsilon


if __name__ == "__main__":
    test_data = {
        1: [0.5, 1, 0, 0, 1],
        2: [0, 1, 1, 1, 1]
    }

    # ~calculate that~
    epsilon = theo_opt_epsilon(test_data)
    print(epsilon)
    
    # implement the algorithms & calculate rergret
    exponential_weights(test_data, epsilon)
    follow_perturbed_leader(test_data)

    # compare the regrets of these two alorithms againist each other
