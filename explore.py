#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

import utils

gifts = pd.read_csv("data/gifts.csv")

plt.ion()

def plot_weights():
  sns.distplot(gifts.Weight)
  plt.show()

def load_solution(file_name):
  return pd.read_csv(file_name).merge(gifts, on="GiftId")

def prepare_and_show_trip_plot():
  plt.colorbar()
  plt.grid()
  plt.tight_layout()
  plt.xlim(-180, 180)
  plt.xlabel("Longitude")
  plt.ylim(-90, 90)
  plt.ylabel("Latitude")
  plt.axis("equal")
  plt.show()

def plot_trips(solution_file):
  trips = pd.read_csv(solution_file).merge(gifts, on="GiftId")
  fig = plt.figure()
  for t in trips.TripId.unique():
    if t % 100 == 0:
      print(str(t) + "... ", end="", flush=True)
    trip = trips[trips.TripId == t]
    plt.scatter(trip.Longitude, trip.Latitude, c=trip.Weight,  alpha=0.8, s=4, linewidths=0)
    plt.plot(trip.Longitude, trip.Latitude, 'k.-', alpha=0.005)
  plt.title("All trips")
  prepare_and_show_trip_plot()

def plot_trip(solution_file, t=None):
  trips = pd.read_csv(solution_file).merge(gifts, on="GiftId")
  fig = plt.figure()
  if t is None:
    trip_ids = trips.TripId.unique()
    t = trip_ids[np.random.randint(len(trip_ids))]
  trip = trips[trips.TripId == t]
  trip = pd.concat([pd.DataFrame([{"GiftId": 0, "Latitude": utils.NORTH_POLE[0], "Longitude": utils.NORTH_POLE[1], "TripId": t, "Weight": 0}]), trip])
  plt.scatter(trip.Longitude, trip.Latitude, c=trip.Weight,  alpha=0.8, s=10, linewidths=4)
  plt.plot(trip.Longitude, trip.Latitude, 'k.-', alpha=0.3)
  plt.title("Trip {} ({})".format(t, solution_file))
  prepare_and_show_trip_plot()

def print_stats(file_name=None, df=None, plots=False):
  if file_name is not None:
    df = pd.read_csv(file_name).merge(gifts, on="GiftId")
  if df is None:
    print("Need to specify either file name or df")

  score = utils.weighted_reindeer_weariness(df)
  trip_sizes = df.groupby("TripId").size()
  trips = df.TripId.unique()
  weights = np.array([df[df.TripId == trip].Weight.sum() for trip in trips])
  costs = np.array([utils.weighted_trip_length(df[df.TripId == trip][["Longitude", "Latitude"]], df[df.TripId == trip].Weight) for trip in trips])
  efficiencies = weights / costs
  print("Score: {:.5f}B for {} trips".format(score / 1e9, len(trip_sizes)))
  print("Trip sizes: min/median/max:\t\t{:>6.2f}\t{:>6.2f}\t{:>7.2f};\t{:>6.2f}+-{:>9.2f}".format(
    trip_sizes.min(), trip_sizes.median(), trip_sizes.max(), trip_sizes.mean(), trip_sizes.std()**2))
  print("Costs per trip: min/median/max [M]:\t{:>6.2f}\t{:>6.2f}\t{:>7.2f};\t{:>6.2f}+-{:>9.2f}".format(
    costs.min()/1e6, np.median(costs)/1e6, costs.max()/1e6, costs.mean()/1e6, (costs.std()/1e6)**2))
  print("Weights per trip: min/median/max:\t{:>6.2f}\t{:>6.2f}\t{:>7.2f};\t{:>6.2f}+-{:>9.2f}".format(
    weights.min(), np.median(weights), weights.max(), weights.mean(), (weights.std())**2))
  print("Efficiencies per trip: min/median/max:\t{:>6.2f}\t{:>6.2f}\t{:>7.2f};\t{:>6.2f}+-{:>9.2f}".format(
    efficiencies.min()*1e6, np.median(efficiencies)*1e6, efficiencies.max()*1e6, efficiencies.mean()*1e6, (efficiencies.std()*1e6)**2))

  if plots:
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].hist(weights, bins=100)
    axes[0, 0].set_title("Weights")
    axes[0, 1].hist(costs, bins=100)
    axes[0, 1].set_title("Costs")
    axes[1, 0].hist(efficiencies, bins=100)
    axes[1, 0].set_title("Efficiencies")
    axes[1, 1].hist(trip_sizes, bins=100)
    axes[1, 1].set_title("Trip sizes")
    if file_name is not None:
      fig.suptitle("Stats for {}".format(file_name))

def plot_metrics(file_name, cumulative=True, cost_factor=1e4, temperature_factor=1e3):
  with open(file_name, "rb") as fh:
    (iterations, interval, temperatures, good_solutions, accepted_solutions, rejected_solutions, costs) = pickle.load(fh)
  if cumulative:
    costs = np.cumsum(costs)
    good_solutions = np.cumsum(good_solutions)
    accepted_solutions = np.cumsum(accepted_solutions)
    rejected_solutions = np.cumsum(rejected_solutions)
  x = np.linspace(0, iterations, int(iterations/interval))
  plt.plot(x, [cost / cost_factor for cost in costs], label="Cost change [{}]".format(cost_factor), color="blue")
  plt.plot(x, good_solutions, label="Good solutions", color="green")
  plt.plot(x, accepted_solutions, label="Accepted solutions", color="orange")
  plt.plot(x, rejected_solutions, label="Rejected solutions", color="red")
  plt.plot(x, [temp / temperature_factor for temp in temperatures], label="Temperature [{}]".format(temperature_factor), color="gray")
  plt.legend()
  plt.grid()
  plt.tight_layout()
  plt.title("Metrics at checkpoint '{}'".format(file_name))

