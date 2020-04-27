import csv
import numpy as np
from scipy.stats import geom
import math

## --------------- NOTES ------------------ ##
# The idea here is to compare the regret of the two algorithms
# Must implement the algorithms on random data
# Calculate thier regrets (REGRET = 1/n * [OPT - ALG])
# Tune learning rates & compare


def exponential_weights(test_data, epsilon, h):
    action1 = test_data[1]
    action2 = test_data[2]
    total_payoff = 0

    for r in range(len(test_data[1])):
        # calculate the probability of chosing every action in round r 
        probabilities, round_payoffs = get_probabilities(r, epsilon, h, test_data)
        print(probabilities, round_payoffs)
        # chose action for round r with probabilities
        action_payoff = np.random.choice(round_payoffs, p=probabilities)
        print(action_payoff)
        total_payoff += action_payoff

    # regret is 2 * h sqrt(ln(k) / h)
    # learning rate is sqrt(ln(k) / h)

    print("EW TOTAL PAYOFF", total_payoff)
    return total_payoff

def get_probabilities(r, e, h, test_data):
    hindsight_payoffs = []
    total_payoff = 0
    probabilities = []
    curr_payoffs = []
    
    if r == 0 :
        for action in range(len(test_data)):
            action_payoff = test_data[action + 1][0]
            curr_payoffs.append(action_payoff)
            hindsight_payoff = 0
            hindsight_payoffs.append(hindsight_payoff)
        return [0.5, 0.5], curr_payoffs
    else:
        for action in range(len(test_data)):
            action_payoff = test_data[action + 1][r]
            curr_payoffs.append(action_payoff)
            hindsight_payoff = sum(test_data[action + 1][:r])
            hindsight_payoffs.append((1+e) ** (hindsight_payoff/h))
        total_payoff = sum(hindsight_payoffs)

        for action in range(len(test_data)):
            probabilities.append(hindsight_payoffs[action]/total_payoff)

        return probabilities, curr_payoffs


def follow_perturbed_leader(test_data, epsilon):
    # get values for each action at every round
    action1 = test_data[1].copy()
    action2 = test_data[2].copy()

    # generate hallucinations 
    hallucinations = geom.rvs(epsilon, size=len(test_data))
    print(hallucinations)
    
    # add a halicination at round 0
    action1.insert(0, hallucinations[0])
    action2.insert(0, hallucinations[1])
    
    # loop through and at each day choose the payoff of the BIH action
    # key here is that we take into account the round 0 hallucinations
    ftpl = 0
    for idx in range(len(action1)):
        if idx == 0:
            continue
        else:
            bih1, bih2 = best_in_hindsight(action1, action2, idx)
            if bih2 > bih1:
                ftpl += action2[idx]
                print("action 2")
            else:
                ftpl += action1[idx]
                print("action 1")
    
    # now we calculated payoff for FTPL
    # can further compare with OPT to get regret
    print('FTPL TOTAL PAYOFF',ftpl)
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

def calculate_regret(test_data, alg):
    # calculate OPT
    action_bihs = []
    for action in test_data:
        print(test_data[action])
        action_bihs.append(sum(test_data[action]))
    print("action_bihs", action_bihs)
    best_bih = max(action_bihs)
    
    print(best_bih)
    regret = (best_bih - alg) / len(test_data[1])
    
    return regret


if __name__ == "__main__":
    test_data = {
        1: [0.5, 1, 0, 0, 1],
        2: [0, 1, 1, 1, 1]
    }

    # ~calculate that~
    epsilon = theo_opt_epsilon(test_data)
    print(epsilon)
    h = 1
    
    # implement the algorithms & calculate rergret
    ew = exponential_weights(test_data, epsilon, h)
    ftpl = follow_perturbed_leader(test_data, epsilon)
    
    # compare the regrets of these two alorithms againist each other
    ew_regret = calculate_regret(test_data, ew)
    print('ew regret', ew_regret)
    ftpl_regret = calculate_regret(test_data, ftpl)
    print('ftpl regret', ftpl_regret)
