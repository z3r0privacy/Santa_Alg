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

  """
  def _get_cost_of_adding_gift_at_end(gift, cum_distance, previous_location):
    pass

  def _get_cost_of_removing_gift_in_beginning(gift, remaining_distance):
    pass

  def _find_best_split_index(self, trip):
    cost_of_old_trip = utils.weighted_trip_length(pd.DataFrame(trip)[utils.LOCATION], pd.DataFrame(trip)[utils.WEIGHT])

    # initialize by splitting before gift 2 (to get one 1-gift and one n-1-gift trip)
    best_index = 1
    previous_location = utils.get_location(trip[0])
    first_trip_distance = utils.distance(utils.NORTH_POLE, previous_location)
    second_trip_distance = utils.distance(utils.NORTH_POLE, utils.get_location(trip[len(trip)-1]))
    for k in range(best_index, len(trip)-1):
      second_trip_distance += utils.distance(utils.get_location(trip[k]), utils.get_location(trip[k+1]))
    # print(first_trip_distance, second_trip_distance)
    print("\t\t",first_trip_distance + second_trip_distance + utils.distance(previous_location, utils.get_location(trip[best_index])))

    cost_trip_1 = first_trip_distance * (trip[0][utils.WEIGHT] + 2 * utils.SLEIGH_WEIGHT) # carry the sleigh there and back
    cost_trip_2 = utils.weighted_trip_length(trip[best_index:, utils.LOCATION], trip[best_index:, utils.WEIGHT])
    minimum_cost = cost_trip_1 + cost_trip_2 - cost_of_old_trip

    print("prev", previous_location)
    for i in range(2, len(trip)):
      print("i", i)
      current_gift = trip[i]
      next_location = utils.get_location(trip[i+1]) if i < len(trip)-1 else utils.NORTH_POLE

      # we now "move" gift i from the second to the first trip
      print("gift", current_gift)
      old_previous_location = previous_location
      previous_location = utils.get_location(trip[i-1])
      first_trip_distance += utils.distance(old_previous_location, previous_location)
      second_trip_distance -= utils.distance(previous_location, current_gift[utils.LOCATION])

      print("prev", previous_location)
      print("next", next_location)

      print("\t\t",first_trip_distance + second_trip_distance + utils.distance(previous_location, utils.get_location(current_gift)))

      cost_of_adding = self._get_cost_of_adding_gift_at_end(current_gift, first_trip_distance, previous_location)
      cost_of_removing = self._get_cost_of_removing_gift_in_beginning(current_gift, next_location, second_trip_distance)
      if i == 5:
        raise ValueError("NOT IMPLEMENTED")
    """

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

    new_trip = trip[self.index_to_split:]
    new_trip[:, utils.TRIP] = len(self.trips) + 1
    self.trips[self.trip_to_split] = trip[:self.index_to_split]
    self.trips.append(new_trip)

