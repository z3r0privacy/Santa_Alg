#!/usr/bin/env python


import numpy as np
import pandas as pd

import utils
from neighbor import Neighbor


class SplitOneTripIntoTwoNeighbor(Neighbor):
  def __init__(self, trips):
    self.trips = trips
    self.trip_to_split = np.random.randint(len(trips))
    while len(self.trips[self.trip_to_split]) < 2:
      self.trip_to_split = np.random.randint(len(trips))
    self.index_to_split = None
    super(SplitOneTripIntoTwoNeighbor, self).__init__()

  def __str__(self):
    return "split-{}-at-{}".format(self.trip_to_split, self.index_to_split)

  def _find_best_split_index(self, trip):
    minimum_cost = np.finfo(np.float64).max
    best_index = None

    # don't split before first item
    for i in range(1, len(trip)):
      first_trip = trip[:i]
      second_trip = trip[i:]
      cost_first_trip = utils.weighted_trip_length(first_trip[:, utils.LOCATION],first_trip[:, utils.WEIGHT])
      cost_second_trip = utils.weighted_trip_length(second_trip[:, utils.LOCATION],second_trip[:, utils.WEIGHT])
      current_cost = cost_first_trip + cost_second_trip
      if current_cost < minimum_cost:
        minimum_cost = current_cost
        best_index = i
    return best_index, minimum_cost

  def cost_delta(self):
    if self.cost is not None:
      return self.cost

    trip = self.trips[self.trip_to_split]
    cost_of_old_trip = utils.weighted_trip_length(trip[:, utils.LOCATION], trip[:, utils.WEIGHT])

    # find split index with minimum cost
    self.index_to_split, cost_of_split = self._find_best_split_index(trip)

    self.cost = cost_of_split - cost_of_old_trip
    return self.cost

  def apply(self):
    # self.log.debug("Applying {}".format(self))

    trip = self.trips[self.trip_to_split]

    if self.VERIFY_COST_DELTA:
      old = utils.weighted_trip_length(trip[:, utils.LOCATION], trip[:, utils.WEIGHT])

    new_trip = trip[self.index_to_split:]
    existing_trips = [t[0, utils.TRIP] for t in self.trips]
    new_trip_id = np.max(existing_trips) + 1
    new_trip[:, utils.TRIP] = new_trip_id
    self.trips[self.trip_to_split] = trip[:self.index_to_split]
    self.trips.append(new_trip)

    if self.VERIFY_COST_DELTA:
      new = utils.weighted_trip_length(self.trips[self.trip_to_split][:, utils.LOCATION], self.trips[self.trip_to_split][:, utils.WEIGHT]) + \
          utils.weighted_trip_length(new_trip[:, utils.LOCATION], new_trip[:, utils.WEIGHT])
      utils.verify_costs_are_equal(self.cost_delta(), new-old)


class OptimalHorizontalTripSplitNeighbor(Neighbor):
  def __init__(self, trips):
    self.trips = trips
    self.trip_to_split = np.random.randint(len(trips))
    while len(self.trips[self.trip_to_split]) < 4:
      self.trip_to_split = np.random.randint(len(trips))
    self.longitude_to_split = None
    super(OptimalHorizontalTripSplitNeighbor, self).__init__()

  def __str__(self):
    return "hsplit-{}-at-{}".format(self.trip_to_split, self.longitude_to_split)

  def cost_delta(self):
    if self.cost is not None:
      return self.cost

    trip = self.trips[self.trip_to_split]
    cost_of_old_trip = utils.weighted_trip_length(trip[:, utils.LOCATION], trip[:, utils.WEIGHT])

    # check splitting in the middle third of longitudes
    longitudes = np.sort(trip[:, utils.LON][:])[int(len(trip)/3.0):int(len(trip)*2.0/3)]

    minimum_cost = np.finfo(np.float64).max

    for i, lon in enumerate(longitudes):
      # split trips and sort by LAT descending
      trip_1 = trip[trip[:, utils.LON] < lon]
      trip_1 = trip_1[trip_1[:, utils.LAT].argsort()[::-1]]
      trip_2 = trip[trip[:, utils.LON] >= lon]
      trip_2 = trip_2[trip_2[:, utils.LAT].argsort()[::-1]]

      if len(trip_1) * len(trip_2) == 0:
        # don't split here if one of the resulting trips is empty
        continue

      cost_2_1 = utils.weighted_trip_length(trip_1[:, utils.LOCATION], trip_1[:, utils.WEIGHT])
      cost_2_2 = utils.weighted_trip_length(trip_2[:, utils.LOCATION], trip_2[:, utils.WEIGHT])
      if cost_2_1 + cost_2_2 < minimum_cost:
        minimum_cost = cost_2_1 + cost_2_2
        self.longitude_to_split = lon

    self.cost = minimum_cost - cost_of_old_trip
    return self.cost

  def apply(self):
    # self.log.debug("Applying {}".format(self))

    trip = self.trips[self.trip_to_split]

    if self.VERIFY_COST_DELTA:
      old = utils.weighted_trip_length(trip[:, utils.LOCATION], trip[:, utils.WEIGHT])

    trip_1 = trip[trip[:, utils.LON] < self.longitude_to_split]
    trip_1 = trip_1[trip_1[:, utils.LAT].argsort()[::-1]]
    trip_2 = trip[trip[:, utils.LON] >= self.longitude_to_split]
    trip_2 = trip_2[trip_2[:, utils.LAT].argsort()[::-1]]

    existing_trips = [t[0, utils.TRIP] for t in self.trips]
    new_trip_id = np.max(existing_trips) + 1
    trip_2[:, utils.TRIP] = new_trip_id
    self.trips[self.trip_to_split] = trip_1
    self.trips.append(trip_2)

    if self.VERIFY_COST_DELTA:
      new = utils.weighted_trip_length(self.trips[self.trip_to_split][:, utils.LOCATION], self.trips[self.trip_to_split][:, utils.WEIGHT]) + \
          utils.weighted_trip_length(trip_2[:, utils.LOCATION], trip_2[:, utils.WEIGHT])
      utils.verify_costs_are_equal(self.cost_delta(), new-old)


class OptimalVerticalTripSplitNeighbor(Neighbor):
  def __init__(self, trips):
    self.trips = trips
    self.trip_to_split = np.random.randint(len(trips))
    while len(self.trips[self.trip_to_split]) < 4:
      self.trip_to_split = np.random.randint(len(trips))
    self.latitude_to_split = None
    super(OptimalVerticalTripSplitNeighbor, self).__init__()

  def __str__(self):
    return "vsplit-{}-at-{}".format(self.trip_to_split, self.latitude_to_split)

  def cost_delta(self):
    if self.cost is not None:
      return self.cost

    trip = self.trips[self.trip_to_split]
    cost_of_old_trip = utils.weighted_trip_length(trip[:, utils.LOCATION], trip[:, utils.WEIGHT])

    # check splitting in the middle third of latitudes
    latitudes = np.sort(trip[:, utils.LAT][:])[int(len(trip)/3.0):int(len(trip)*2.0/3)]

    minimum_cost = np.finfo(np.float64).max

    for i, lat in enumerate(latitudes):
      # split trips and sort by LON descending
      trip_1 = trip[trip[:, utils.LAT] < lat]
      trip_1 = trip_1[trip_1[:, utils.LON].argsort()[::-1]]
      trip_2 = trip[trip[:, utils.LAT] >= lat]
      trip_2 = trip_2[trip_2[:, utils.LON].argsort()[::-1]]

      if len(trip_1) * len(trip_2) == 0:
        # don't split here if one of the resulting trips is empty
        continue

      cost_2_1 = utils.weighted_trip_length(trip_1[:, utils.LOCATION], trip_1[:, utils.WEIGHT])
      cost_2_2 = utils.weighted_trip_length(trip_2[:, utils.LOCATION], trip_2[:, utils.WEIGHT])
      if cost_2_1 + cost_2_2 < minimum_cost:
        minimum_cost = cost_2_1 + cost_2_2
        self.latitude_to_split = lat

    self.cost = minimum_cost - cost_of_old_trip
    return self.cost

  def apply(self):
    # self.log.debug("Applying {}".format(self))

    trip = self.trips[self.trip_to_split]

    if self.VERIFY_COST_DELTA:
      old = utils.weighted_trip_length(trip[:, utils.LOCATION], trip[:, utils.WEIGHT])

    trip_1 = trip[trip[:, utils.LAT] < self.latitude_to_split]
    trip_1 = trip_1[trip_1[:, utils.LON].argsort()[::-1]]
    trip_2 = trip[trip[:, utils.LAT] >= self.latitude_to_split]
    trip_2 = trip_2[trip_2[:, utils.LON].argsort()[::-1]]

    existing_trips = [t[0, utils.TRIP] for t in self.trips]
    new_trip_id = np.max(existing_trips) + 1
    trip_2[:, utils.TRIP] = new_trip_id
    self.trips[self.trip_to_split] = trip_1
    self.trips.append(trip_2)

    if self.VERIFY_COST_DELTA:
      new = utils.weighted_trip_length(self.trips[self.trip_to_split][:, utils.LOCATION], self.trips[self.trip_to_split][:, utils.WEIGHT]) + \
          utils.weighted_trip_length(trip_2[:, utils.LOCATION], trip_2[:, utils.WEIGHT])
      utils.verify_costs_are_equal(self.cost_delta(), new-old)

