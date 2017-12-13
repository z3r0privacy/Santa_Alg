#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils

gifts = pd.read_csv("data/gifts.csv")

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

def plot_trip(solution_file, t=123):
  trips = pd.read_csv(solution_file).merge(gifts, on="GiftId")
  fig = plt.figure()
  trip = trips[trips.TripId == t]
  trip = pd.concat([pd.DataFrame([{"GiftId": 0, "Latitude": utils.NORTH_POLE[0], "Longitude": utils.NORTH_POLE[1], "TripId": t, "Weight": 0}]), trip])
  plt.scatter(trip.Longitude, trip.Latitude, c=trip.Weight,  alpha=0.8, s=10, linewidths=4)
  plt.plot(trip.Longitude, trip.Latitude, 'k.-', alpha=0.3)
  plt.title("Trip {} ({})".format(t, solution_file))
  prepare_and_show_trip_plot()

