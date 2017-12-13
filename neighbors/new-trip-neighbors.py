#!/usr/bin/env python

import numpy as np

import pandas as pd
import utils
from neighbor import Neighbor
from utils import memoize


class SplitOneTripIntoTwoNeighbor(Neighbor):
  def __init__(self, trips, log):
    # when trip to split isn't specified, select one randomly
    self.trips = trips
    self.trip_to_split = np.random.randint(len(trips))
    while len(self.trips[self.trip_to_split]) < 2:
      self.trip_to_split = np.random.randint(len(trips))
    self.index_to_split = None
    super(SplitOneTripIntoTwoNeighbor, self).__init__(log)

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

  @property
  @memoize
  def cost_delta(self):
    trip = self.trips[self.trip_to_split]
    cost_of_old_trip = utils.weighted_trip_length(trip[:, utils.LOCATION], trip[:, utils.WEIGHT])

    # find split index with minimum cost
    self.index_to_split, cost_of_split = self._find_best_split_index(trip)

    return cost_of_split - cost_of_old_trip

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
      utils.verify_costs_are_equal(self.cost_delta, new-old)

