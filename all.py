import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensortrade.env.default as default
import PyQt5
# from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.actions import BSH
import tensorflow as tf
from tensortrade.agents import DQNAgent

# read data
data = pd.read_csv("../data/others/BTCUSDT-2h-data.csv")
train_data = data[0:int(len(data)*0.7)]   # training
eval_data = data[int(len(data)*0.7):]   #  evaluation
print(len(data)*0.7)

###

# train features
t_features = []
for c in train_data.columns[1:]:
    s = Stream.source(list(train_data[c]), dtype="float").rename(train_data[c].name)
    t_features += [s]

t_op = Stream.select(t_features, lambda s: s.name == "open")
t_hp = Stream.select(t_features, lambda s: s.name == "high")
t_lp = Stream.select(t_features, lambda s: s.name == "low")
t_cp = Stream.select(t_features, lambda s: s.name == "close")

t_oi = Stream.select(t_features, lambda s: s.name == "sumOpenInterest")
t_lsur = Stream.select(t_features, lambda s: s.name == "longShortRatio")
t_vol = Stream.select(t_features, lambda s: s.name == "volume")

t_features_r = [
    # t_oi.pct_change().rename("oid"),
    # t_lsur.pct_change().rename("lsurd"),
    # t_vol.pct_change().rename("vold"),
    t_cp.pct_change().rename("cd")
]

t_feed = DataFeed(t_features_r)
t_feed.compile()
t_exops = ExchangeOptions(commission=0.005)

###

# train portfolio
t_bitstamp = Exchange("bitstamp", service=execute_order,options=t_exops)(
    Stream.source(list(train_data["close"]), dtype="float").rename("USD-BTC")
)
t_cash = Wallet(t_bitstamp, 10000 * USD)
t_asset = Wallet(t_bitstamp, 0 * BTC)
t_portfolio = Portfolio(USD, [
    t_cash,
    t_asset
])

###

# train render
t_renderer_feed = DataFeed([
    Stream.source(list(train_data["date"])).rename("date"),
    Stream.source(list(train_data["open"]), dtype="float8").rename("open"),
    Stream.source(list(train_data["high"]), dtype="float8").rename("high"),
    Stream.source(list(train_data["low"]), dtype="float8").rename("low"),
    Stream.source(list(train_data["close"]), dtype="float8").rename("close"), 
    Stream.source(list(train_data["volume"]), dtype="float8").rename("volume") 
])

###

# train env
t_env = default.create(
    portfolio=t_portfolio,
    action_scheme=BSH(cash=t_cash,asset=t_asset),
    reward_scheme="simple",
    feed=t_feed,
    renderer_feed=t_renderer_feed,
    renderer="screen-log",
    window_size=18
)
# print(train_env.observer.feed.next())

###

# eval features
e_features = []
for c in eval_data.columns[1:]:
    s = Stream.source(list(eval_data[c]), dtype="float").rename(eval_data[c].name)
    e_features += [s]

e_op = Stream.select(e_features, lambda s: s.name == "open")
e_hp = Stream.select(e_features, lambda s: s.name == "high")
e_lp = Stream.select(e_features, lambda s: s.name == "low")
e_cp = Stream.select(e_features, lambda s: s.name == "close")

e_oi = Stream.select(e_features, lambda s: s.name == "sumOpenInterest")
e_lsur = Stream.select(e_features, lambda s: s.name == "longShortRatio")
e_vol = Stream.select(e_features, lambda s: s.name == "volume")

e_features_r = [
    # t_oi.pct_change().rename("oid"),
    # t_lsur.pct_change().rename("lsurd"),
    # t_vol.pct_change().rename("vold"),
    e_cp.pct_change().rename("cd")
]

e_feed = DataFeed(e_features_r)
e_feed.compile()
e_exops = ExchangeOptions(commission=0.005)

###

# eval portfolio
e_bitstamp = Exchange("bitstamp", service=execute_order,options=e_exops)(
    Stream.source(list(eval_data["close"]), dtype="float").rename("USD-BTC")
)
e_cash = Wallet(e_bitstamp, 10000 * USD)
e_asset = Wallet(e_bitstamp, 0 * BTC)
e_portfolio = Portfolio(USD, [
    e_cash,
    e_asset
])

###

# eval render
e_renderer_feed = DataFeed([
    Stream.source(list(eval_data["date"])).rename("date"),
    Stream.source(list(eval_data["open"]), dtype="float8").rename("open"),
    Stream.source(list(eval_data["high"]), dtype="float8").rename("high"),
    Stream.source(list(eval_data["low"]), dtype="float8").rename("low"),
    Stream.source(list(eval_data["close"]), dtype="float8").rename("close"), 
    Stream.source(list(eval_data["volume"]), dtype="float8").rename("volume") 
])

###

# eval env
e_env = default.create(
    portfolio=e_portfolio,
    action_scheme=BSH(cash=e_cash,asset=e_asset),
    reward_scheme="simple",
    feed=e_feed,
    renderer_feed=e_renderer_feed,
    renderer="screen-log",
    window_size=18
)
# print(eval_env.observer.feed.next())

### 

# DQN agent
t_env.reset()
network = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=t_env.observation_space.shape), # window_size * 4
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(t_env.action_space.n, activation="sigmoid"), # 2 actions
    tf.keras.layers.Dense(t_env.action_space.n, activation="softmax")  # 2 actions
    ])
print(network.summary())

agent = DQNAgent(t_env, policy_network=network)

with tf.device("gpu:0"):
    agent.train(n_episodes=30,n_steps=4800,memory_capacity=10000,render_interval=200)

# agent.save("./")

###

# train ouptut
performance = pd.DataFrame.from_dict(t_env.action_scheme.portfolio.performance, orient='index')
performance.plot()

###

# eval output
obs = e_env.reset()
e = []
while True:
    action = agent.get_action(obs)
    obs, rewards, dones, info = e_env.step(action)
    e_env.render()
    e.append(info)
    if dones:
        break
performance = pd.DataFrame.from_dict(e_env.action_scheme.portfolio.performance, orient='index')
performance.plot()