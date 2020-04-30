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
        # chose action for round r with probabilities
        action_payoff = np.random.choice(round_payoffs, p=probabilities)
        total_payoff += action_payoff

    # print('total_payoff', total_payoff)

    # regret is 2 * h sqrt(ln(k) / h)
    # learning rate is sqrt(ln(k) / h)

    # print("EW TOTAL PAYOFF", total_payoff)
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
            else:
                ftpl += action1[idx]
    
    # now we calculated payoff for FTPL
    # can further compare with OPT to get regret
    # print('FTPL TOTAL PAYOFF',ftpl)
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
        action_bihs.append(sum(test_data[action]))
    best_bih = max(action_bihs)
    
    regret = (best_bih - alg) / len(test_data[1])
    
    return regret

def generate_data():
    action1 = []
    action2 = []

    act1_prob = 0.5 
    act2_prob = 0.7

    for _ in range(100):
        action1.append(np.random.choice([1, 0], p=[act1_prob, 1-act1_prob]))
        action2.append(np.random.choice([1, 0], p=[act2_prob, 1-act2_prob]))
    
    test_data = {}
    test_data[1] = action1
    test_data[2] = action2

    return test_data

def empricial_anal(test_data, emp_epsilon, alg_name, h):
    best_e = 0
    best_payoff = 0
    payoff_array = []
    best_regret = np.inf
    regret_array = []

    for e in emp_epsilon:
        e_regrets = []

        for i in range(100):
            if alg_name == "ew":
                payoff = exponential_weights(test_data, e, h)
            else:
                payoff = follow_perturbed_leader(test_data, e)

            e_regrets.append(calculate_regret(test_data, payoff))
  
        # payoff_array.append(payoff)

        avg_regret = np.average(e_regrets)
        regret_array.append(avg_regret)

        if avg_regret < best_regret:
            # best_payoff = payoff
            best_regret = avg_regret
            best_e = e
    
    return best_regret, best_e

def patrice_ava_betting(p_a_data):
    h = 1
    theo_epsilon = theo_opt_epsilon(p_a_data)
    print(theo_epsilon)

    # EW
    ew = exponential_weights(p_a_data, theo_epsilon, h)
    ew_regret = calculate_regret(p_a_data, ew)
    print('P V A EW REGRET', ew_regret)

    # FTPL
    ftpl = follow_perturbed_leader(p_a_data, theo_epsilon)
    ftpl_regret = calculate_regret(p_a_data, ftpl)
    print('P V A FTPL REGRET', ftpl_regret)

    emp_epsilon = np.arange(0.01, 0.99, 0.01)
    h = 1

    ew_payoff, ew_regret, ew_learning = empricial_anal(p_a_data, emp_epsilon, "ew", h)
    ftpl_payoff, ftpl_regret, ftpl_learning = empricial_anal(p_a_data, emp_epsilon, "ftpl", h)

    print('EMP EW REGRET', ew_regret)
    # print('EMP EW PAYOFF', ew_payoff)
    print('EMP EW LEARN RATE', ew_learning)


    print('EMP FTPL REGRET', ftpl_regret)
    # print('EMP FTPL PAYOFF', ftpl_payoff)
    print('EMP FTPL LEARN RATE', ftpl_learning)


if __name__ == "__main__":
    # test_data = generate_data()
    # print(test_data)
    test_data = {
        1: [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0], 
        2: [1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]
    }

    # worst_case_test_data = {
    #     1: [.5, 0, 1, 0, 1, 0],
    #     2: [0, 1, 0, 1, 0, 1]
    # }

    # # TESTING WORST CASE SCENARIO 
    # theo_opt_lr_worst_case = theo_opt_epsilon(worst_case_test_data)
    # print('theoretical optimal learning rate:', theo_opt_lr_worst_case)

    # # EW  
    # ew_worst = exponential_weights(worst_case_test_data, theo_opt_lr_worst_case, 1) 
    # ew_regret_worst = calculate_regret(worst_case_test_data, ew_worst)
    # # print('EW REGRET WORST CASE', ew_regret_worst)

    # # FTPL 
    # ftpl_worst = follow_perturbed_leader(worst_case_test_data, theo_opt_lr_worst_case)
    # ftpl_regret_worst = calculate_regret(worst_case_test_data, ftpl_worst)
    # # print('FTPL REGRET WORST CASE', ftpl_regret_worst)

    # # EMPIRCAL WORST CASE 
    # # learning rate
    # emp_epsilon = np.arange(0.01, 0.99, 0.01)
    # h = 1

    # ew_regret_worst, ew_learning_worst = empricial_anal(worst_case_test_data, emp_epsilon, "ew", 1)
    # ftpl_regret_worst, ftpl_learning_worst = empricial_anal(worst_case_test_data, emp_epsilon, "ftpl", 1)
    
    # # print('EMP EW REGRET WORST CASE', ew_regret_worst)
    # # print('EMP EW PAYOFF', ew_payoff_worst)
    # print('EMP EW LEARN RATE', ew_learning_worst)

    # # print('EMP FTPL REGRET WORST CASE', ftpl_regret_worst)
    # # print('EMP FTPL PAYOFF WORST CASE', ftpl_payoff_worst)
    # print('EMP FTPL LEARN RATE WORST CASE', ftpl_learning_worst)

    # PART 1
    # THEORHETICAL 
    # learning rate
    theo_epsilon = theo_opt_epsilon(test_data)
    print('THEO learning rate', theo_epsilon)
    h = 1

    # EW  
    ew = exponential_weights(test_data, theo_epsilon, h) 
    ew_regret = calculate_regret(test_data, ew)
    print('EW REGRET', ew_regret)

    # FTPL 
    ftpl = follow_perturbed_leader(test_data, theo_epsilon)
    ftpl_regret = calculate_regret(test_data, ftpl)
    print('FTPL REGRET', ftpl_regret)

    # EMPIRCAL 
    # learning rate
    emp_epsilon = np.arange(0.01, 0.99, 0.01)
    h = 1

    ew_regret, ew_learning = empricial_anal(test_data, emp_epsilon, "ew", h)
    ftpl_regret, ftpl_learning = empricial_anal(test_data, emp_epsilon, "ftpl", h)
    
    print('EMP EW REGRET', ew_regret)
    # print('EMP EW PAYOFF', ew_payoff)
    print('EMP EW LEARN RATE', ew_learning)


    print('EMP FTPL REGRET', ftpl_regret)
    # print('EMP FTPL PAYOFF', ftpl_payoff)
    print('EMP FTPL LEARN RATE', ftpl_learning)

    germany (1) and italy (2)
    p_a_data = {
        1: [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        2: [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    }

    patrice_ava_betting(p_a_data)

