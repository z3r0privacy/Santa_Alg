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

def plot_trips(solution_file):
  all_trips = gifts.merge(pd.read_csv(solution_file))

  fig = plt.figure()
  plt.scatter(all_trips['Longitude'].values, all_trips['Latitude'].values, c='k', alpha=0.1, s=1, linewidths=0)
  for t in all_trips.TripId.unique():
      previous_location = utils.NORTH_POLE
      trip = all_trips[all_trips['TripId'] == t]
      i = 0
      for _, gift in trip.iterrows():
          plt.plot([previous_location[1], gift['Longitude']], [previous_location[0], gift['Latitude']],
                  color=plt.cm.copper_r(i/90.), alpha=0.1)
          previous_location = tuple(gift[['Latitude', 'Longitude']])
          i += 1
      plt.scatter(gift['Longitude'], gift['Latitude'], c='k', alpha=0.5, s=20, linewidths=0)

  plt.scatter(gift['Longitude'], gift['Latitude'], c='k', alpha=0.5, s=20, linewidths=0, label='TripEnds')
  plt.legend(loc='upper right')
  plt.grid()
  plt.title('TripOrder')
  plt.tight_layout()


  # fig = plt.figure()
  # plt.scatter(trips['Longitude'].values, trips['Latitude'].values, c='k', alpha=0.1, s=1, linewidths=0)
  # for t in trips.TripId.unique():
  #   previous_location = utils.NORTH_POLE
  #   trip = trips[trips['TripId'] == t]
  #   i = 0
  #   for _, gift in trip.iterrows():
  #     plt.plot([previous_location[1], gift['Longitude']], [previous_location[0], gift['Latitude']],
  #             color=plt.cm.copper_r(i/90.), alpha=0.1)
  #     previous_location = tuple(gift[['Latitude', 'Longitude']])
  #     i += 1
  #   plt.scatter(gift['Longitude'], gift['Latitude'], c='k', alpha=0.5, s=20, linewidths=0)

  # plt.scatter(gift['Longitude'], gift['Latitude'], c='k', alpha=0.5, s=20, linewidths=0, label='TripEnds')
  # plt.legend(loc='upper right')
  # plt.grid()
  # plt.title('TripOrder')
  # plt.tight_layout()
  # plt.show()

  # fig = plt.figure()
  # for t in trips.TripId.unique():
  #   trip = trips[trips.TripId == t]
  #   plt.scatter(trip['Longitude'], trip['Latitude'], c=trip['TripId'],  alpha=0.8, s=8, linewidths=0)
  #   plt.plot(trip['Longitude'], trip['Latitude'], 'k.-', alpha=0.1)

  # plt.colorbar()
  # plt.grid()
  # plt.title('Trips')
  # plt.tight_layout()
  # plt.show()

