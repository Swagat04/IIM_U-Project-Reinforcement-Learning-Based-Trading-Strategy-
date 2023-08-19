# IIM_U-Project-Reinforcement-Learning-Based-Trading-Strategy-
IIM_U Project (Reinforcement Learning Based Trading Strategy)
In this Reinforcement Learning framework for trading strategy, the algorithm takes an action (buy, sell or hold) depending upon the current state of the stock price. The algorithm is trained using Deep Q-Learning framework, to help us predict the best action, based on the current stock prices.

The key components of the RL based framework are :

Agent: Trading agent.

Action: Buy, sell or hold.

Reward function: Realized profit and loss (PnL) is used as the reward function for this case study. The reward depends upon the action:

Sell: Realized profit and loss (sell price - bought price)
Buy: No reward
Hold: No Reward
State: Differences of past stock prices for a given time window is used as the state.

The data used for this case study will be the standard and poor's 500. The link to the data is : https://ca.finance.yahoo.com/quote/%255EGSPC/history?p=%255EGSPC).
