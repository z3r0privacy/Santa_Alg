#!/usr/bin/env python

import numpy as np

import pandas as pd
import utils
from neighbor import Neighbor
from utils import memoize

# TODO: More two-trip neighbors like swapping multiple elements

class SwapGiftsAcrossTripsNeighbor(Neighbor):
  def __init__(self, trips, log):
    # when trips/gifts to swap aren't specified, select them randomly
    self.trips = trips
    self.first_trip = np.random.randint(len(trips))
    self.second_trip = np.random.randint(len(trips))
    while self.first_trip == self.second_trip:
      self.second_trip = np.random.randint(len(trips))
    self.first_gift = np.random.randint(len(self.trips[self.first_trip]))
    self.second_gift = self._get_valid_swappee()
    while self.second_gift is None:
      self.first_gift = np.random.randint(len(self.trips[self.first_trip]))
      self.second_gift = self._get_valid_swappee()
    self.first_trip_insertion_index = None
    self.second_trip_insertion_index = None
    super(SwapGiftsAcrossTripsNeighbor, self).__init__(log)

  def __str__(self):
    return "swap-{}:{}-{}:{}".format(self.first_trip, self.first_gift, self.second_trip, self.second_gift)

  def _get_valid_swappee(self):
    weight_of_first_gift = self.trips[self.first_trip][self.first_gift][utils.WEIGHT]
    first_weight = np.sum(self.trips[self.first_trip][:, utils.WEIGHT])
    second_weight = np.sum(self.trips[self.second_trip][:, utils.WEIGHT])

    for gift in np.random.permutation(len(self.trips[self.second_trip])):
      weight_of_second_gift = self.trips[self.second_trip][gift][utils.WEIGHT]
      if (first_weight - weight_of_first_gift + weight_of_second_gift <= utils.WEIGHT_LIMIT and
          second_weight + weight_of_first_gift - weight_of_second_gift <= utils.WEIGHT_LIMIT):
        return gift

  def _find_best_insertion_index(self, trip, gift, index_to_be_removed):
    minimum_cost = np.finfo(np.float64).max
    best_index = None
    for i, row in enumerate(trip):
      if i == index_to_be_removed or i+1 == index_to_be_removed:
        # don't compute insertion where we're about to remove
        continue

      # note: we evaluate inserting before the current node - that means we won't try to insert
      # in the very end of the tour

      # we need to consider the distance from NP to the first gift - unless we're evaluating the first gift
      distance = 0 if i == 0 else utils.distance(utils.NORTH_POLE, tuple(trip[0][[utils.LAT, utils.LON]]))
      # add distances up until the one before the current
      for k in range(i-1):
        distance += utils.distance(
            tuple(trip[k][[utils.LAT, utils.LON]]),
            tuple(trip[k+1][[utils.LAT, utils.LON]])
            )
      cost_to_carry_gift = distance * gift[utils.WEIGHT]

      previous_location = tuple(trip[i-1][[utils.LAT, utils.LON]]) if i > -0 else utils.NORTH_POLE
      location_of_current = tuple(row[[utils.LAT, utils.LON]])
      cum_weight = np.sum(trip[i:][:, utils.WEIGHT]) + utils.SLEIGH_WEIGHT + gift[utils.WEIGHT]
      cost_to_move_here = self.get_cost_of_tour_of_three(
          previous_location, tuple(gift[[utils.LAT, utils.LON]]), location_of_current,
          cum_weight, gift[utils.WEIGHT])

      cost_for_old_path = utils.distance(location_of_current, previous_location) * (cum_weight - gift[utils.WEIGHT])

      cost_to_insert_here = cost_to_carry_gift + cost_to_move_here - cost_for_old_path

      if cost_to_insert_here < minimum_cost:
        minimum_cost = cost_to_insert_here
        best_index = i

    return best_index, minimum_cost

  def _cost_to_remove_gift(self, trip, index_to_be_removed):
    gift_to_remove = trip[index_to_be_removed]
    i = index_to_be_removed

    # we need to consider the distance from NP to the first gift - unless we're evaluating the first gift
    distance = 0 if i == 0 else utils.distance(utils.NORTH_POLE, tuple(trip[0][[utils.LAT, utils.LON]]))
    # add distances up until the one before the current
    for k in range(i-1):
      distance += utils.distance(
          tuple(trip[k][[utils.LAT, utils.LON]]),
          tuple(trip[k+1][[utils.LAT, utils.LON]])
          )
    cost_to_not_carry_gift = distance * -gift_to_remove[utils.WEIGHT]

    previous_location = tuple(trip[i-1][[utils.LAT, utils.LON]]) if i > -0 else utils.NORTH_POLE
    next_location = tuple(trip[i+1][[utils.LAT, utils.LON]]) if i < len(trip)-1 else utils.NORTH_POLE
    location_of_current = tuple(gift_to_remove[[utils.LAT, utils.LON]])
    cum_weight = np.sum(trip[i:][:, utils.WEIGHT]) + utils.SLEIGH_WEIGHT # + gift_to_remove[utils.WEIGHT]
    cost_of_old_tour = self.get_cost_of_tour_of_three(
        previous_location, location_of_current, next_location,
        cum_weight, gift_to_remove[utils.WEIGHT])

    cost_for_new_path = utils.distance(previous_location, next_location) * (cum_weight - gift_to_remove[utils.WEIGHT])

    total_cost = cost_to_not_carry_gift - cost_of_old_tour + cost_for_new_path

    return total_cost


  @property
  @memoize
  def cost_delta(self):
    # find insertion indexes with minimum cost
    self.first_trip_insertion_index, cost_to_insert_first = self._find_best_insertion_index(
        self.trips[self.first_trip], self.trips[self.second_trip][self.second_gift], self.first_gift)
    self.second_trip_insertion_index, cost_to_insert_second = self._find_best_insertion_index(
        self.trips[self.second_trip], self.trips[self.first_trip][self.first_gift], self.second_gift)

    # update temporary trips with new insertion to accurately calculate the cost of deletion
    temporary_first_trip = self.trips[self.first_trip]
    temporary_first_trip = np.insert(temporary_first_trip, self.first_trip_insertion_index, self.trips[self.second_trip][self.second_gift], axis=0)
    temporary_second_trip = self.trips[self.second_trip]
    temporary_second_trip = np.insert(temporary_second_trip, self.second_trip_insertion_index, self.trips[self.first_trip][self.first_gift], axis=0)

    # calculate (negative) cost of deletion
    cost_to_remove_first = self._cost_to_remove_gift(temporary_first_trip, self.first_gift if self.first_gift < self.first_trip_insertion_index else self.first_gift + 1)
    cost_to_remove_second = self._cost_to_remove_gift(temporary_second_trip, self.second_gift if self.second_gift < self.second_trip_insertion_index else self.second_gift + 1)

    total_cost = cost_to_insert_first + cost_to_insert_second + cost_to_remove_first + cost_to_remove_second

    return total_cost

  def apply(self):
    # self.log.debug("Applying {}".format(self))

    first_trip = self.trips[self.first_trip]
    second_trip = self.trips[self.second_trip]

    # extract insertees now (before they're removed) and update their trip assignment
    first_gift_row = first_trip[self.first_gift]
    first_gift_row[utils.TRIP] = self.second_trip+1
    second_gift_row = second_trip[self.second_gift]
    second_gift_row[utils.TRIP] = self.first_trip+1

    # update first trip
    first_trip = np.insert(first_trip, self.first_trip_insertion_index, second_gift_row, axis=0)

    index_to_remove = self.first_gift if self.first_gift < self.first_trip_insertion_index else self.first_gift + 1
    first_trip = np.delete(first_trip, index_to_remove, axis=0)
    self.trips[self.first_trip] = first_trip

    # update second trip
    second_trip = np.insert(second_trip, self.second_trip_insertion_index, first_gift_row, axis=0)

    index_to_remove = self.second_gift if self.second_gift < self.second_trip_insertion_index else self.second_gift + 1
    second_trip = np.delete(second_trip, index_to_remove, axis=0)
    self.trips[self.second_trip] = second_trip

