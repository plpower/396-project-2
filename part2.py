import csv
import matplotlib.pyplot as plt
import numpy as np

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

    for j in range(my_value + 1):
        test_data[j] = []

    return test_data


def exponential_weights(test_data, epsilon, h, opponent_bids):
    total_payoff = 0

    for r in range(my_value + 1):
        # (1) GENERATE A RANDOM COMPETING BID FROM BID_DATA
        opponent_bid = np.random.choice(opponent_bids)

        # (2) UPDATE TEST_DATA @ CURRENT ROUND W WHICH ACTIONS ARE WINNERS
        for key, val in test_data.items():
            if key > opponent_bid:
                val.append(1)
            else:
                val.append(0)


        # (3) CALCULATE THE PROBABILITY
        # calculate the probability of chosing every action in round r
        probabilities, round_payoffs = get_probabilities(
            r, epsilon, h, test_data)
        # chose action for round r with probabilities
        action_payoff = np.random.choice(round_payoffs, p=probabilities)
        print(action_payoff)
        total_payoff += action_payoff
    

    print('total_payoff', total_payoff)
    # print("EW TOTAL PAYOFF", total_payoff)
    return total_payoff


def get_probabilities(r, e, h, test_data):
    hindsight_payoffs = []
    total_payoff = 0
    probabilities = []
    curr_payoffs = []

    number_of_actions = len(test_data)

    if r == 0:
        for action in range(number_of_actions):
            action_payoff = test_data[action][0]
            curr_payoffs.append(action_payoff)
            hindsight_payoff = 0
            hindsight_payoffs.append(hindsight_payoff)
            probs = [(1/number_of_actions) for _ in range(number_of_actions)]
        return probs, curr_payoffs
    else:
        for action in range(number_of_actions):
            action_payoff = test_data[action + 1][r]
            curr_payoffs.append(action_payoff)
            hindsight_payoff = sum(test_data[action + 1][:r])
            hindsight_payoffs.append((1+e) ** (hindsight_payoff/h))
        total_payoff = sum(hindsight_payoffs)

        for action in range(number_of_actions):
            probabilities.append(hindsight_payoffs[action]/total_payoff)

        return probabilities, curr_payoffs



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
    h = 1
    epsilon = 0.3

    bid_actions = generate_test_data(my_value)

    # WE NEED TO CALCULATE EPSILON !!!
    total_payoff = exponential_weights(bid_actions, epsilon, h, opponent_bids)
