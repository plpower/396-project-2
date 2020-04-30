import csv
import matplotlib.pyplot as plt
import numpy as np
import math

data = open('bid_data.csv')
csv_data = csv.reader(data)
# given_values = [32.1, 50, 31, 81.4]
given_values = list(range(1, 101, 1))
our_bids = [30, 45, 30, 75]

# --------------- NOTES -------------- #
# data space is each bid
# at each round, we update the "hindsight list" of which bids would have been winners
# use the alg that we decide on (FTPL or EW)

def generate_test_data(my_value):
    # start with the example of our bid of 45 with a given value of 50
    # discretize the bid space to k bids between 0 and v
    # action j corresponds to bidding bj

    test_data = {}

    for j in np.arange(0, my_value + 1, 2):
        test_data[j] = []

    return test_data


def exponential_weights(test_data, epsilon, h, opponent_bids):
    total_payoff = 0
    all_op_bids = []

    for r in range(my_value + 1):
        # (1) GENERATE A RANDOM COMPETING BID FROM BID_DATA
        opponent_bid = np.random.choice(opponent_bids)
        all_op_bids.append(opponent_bid)

        # (2) UPDATE TEST_DATA @ CURRENT ROUND W WHICH ACTIONS ARE WINNERS
        for key, val in test_data.items():
            if key > opponent_bid:
                val.append(key-opponent_bid)
            else:
                val.append(0)


        # (3) CALCULATE THE PROBABILITY
        # calculate the probability of chosing every action in round r
        probabilities, round_payoffs = get_probabilities(
            r, epsilon, h, test_data)
        # chose action for round r with probabilities
        if r == 0:
            action_payoff = np.random.choice(round_payoffs)
        else:
            action_payoff = np.random.choice(round_payoffs, p=probabilities)
        total_payoff += action_payoff
    
    # print(test_data)
    # print(all_op_bids)

    # print('total_payoff', total_payoff)
    # print("EW TOTAL PAYOFF", total_payoff)
    return total_payoff, all_op_bids, test_data


def get_probabilities(r, e, h, test_data):
    hindsight_payoffs = []
    total_payoff = 0
    probabilities = []
    curr_payoffs = []

    number_of_actions = 26

    # print(len(np.arange(0, my_value + 0.01, 0.01)))
    if r == 0:
        for action in np.arange(0, my_value + 1, 2):
            action_payoff = test_data[action][0]
            curr_payoffs.append(action_payoff)
            hindsight_payoff = 0
            hindsight_payoffs.append(hindsight_payoff)
            # probs = [(1/number_of_actions) for _ in range(5001)]
            # # print(len(probs))
            # print(np.sum(probs))
            probs = [1 for _ in range(26)]
        return probs, curr_payoffs
    else:
        for action in np.arange(0, my_value + 1, 2):
            action_payoff = test_data[action][r]
            curr_payoffs.append(action_payoff)
            hindsight_payoff = sum(test_data[action][:r])
            hindsight_payoffs.append((1+e) ** (hindsight_payoff/h))
        total_payoff = sum(hindsight_payoffs)

        for action in range(26):
            probabilities.append(hindsight_payoffs[action]/total_payoff)

        return probabilities, curr_payoffs

def theo_opt_epsilon(test_data):
    k = 25
    n = 50
    epsilon = math.sqrt(np.log(k)/n)
    
    return epsilon

def empricial_anal(test_data, emp_epsilon, h, opponent_bids):
    best_e = 0
    best_payoff = 0
    payoff_array = []
    best_regret = np.inf
    regret_array = []

    for e in emp_epsilon:
        e_regrets = []

        for i in range(10):
            payoff, _, test_data = exponential_weights(test_data, e, h, opponent_bids)

            e_regrets.append(calculate_regret(test_data, payoff))
            print("here")
  
        # payoff_array.append(payoff)

        avg_regret = np.average(e_regrets)
        regret_array.append(avg_regret)

        if avg_regret < best_regret:
            # best_payoff = payoff
            best_regret = avg_regret
            best_e = e
    
    return best_regret, best_e

def calculate_regret(test_data, alg):
    # calculate OPT
    action_bihs = []
    for action in test_data:
        action_bihs.append(sum(test_data[action]))
    best_bih = max(action_bihs)
    
    regret = (best_bih - alg) / len(test_data[2])
    
    return regret


if __name__ == "__main__":

    # transforming data
    opponent_values, opponent_bids = [], []
    for row in csv_data:
        opponent_values.append(float(row[0]))
        opponent_bids.append(float(row[1]))
    opponent_values = np.asarray(opponent_values)
    opponent_bids = np.asarray(opponent_bids)

    my_value = 50
    my_bid = 45
    h = my_value

    bid_actions = generate_test_data(my_value)
    epsilon = theo_opt_epsilon(bid_actions)
    print(epsilon)
    # WE NEED TO CALCULATE EPSILON !!!
    avg_regret = []
    for i in range(100):
        total_payoff, all_op_bids, test_data = exponential_weights(bid_actions, epsilon, h, opponent_bids)
        regret = calculate_regret(test_data, total_payoff)
        avg_regret.append(regret)

    print('theo payoff', np.average(total_payoff))
    print("THEO", np.average(avg_regret))

    # best_sum = 0
    # for key, vals in test_data.items():
    #     if best_sum < sum(test_data[key]):
    #         best_sum = sum(test_data[key])
    # print("OPT_payoff", best_sum)

    # # regret = calculate_regret(test_data, total_payoff)
    # # print("THEO regret", regret)

    emp_epsilon = np.arange(0.01, 0.99, 0.01)
    best_regret, best_e = empricial_anal(bid_actions, emp_epsilon, h, opponent_bids)
    print("EMP best regret", best_regret)
    print("EMP learning rate", best_e)


