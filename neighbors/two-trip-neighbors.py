#!/usr/bin/env python

import numpy as np

import pandas as pd
import utils
from neighbor import Neighbor
from utils import memoize


class MoveGiftToAnotherTripNeighbor(Neighbor):
  def __init__(self, trips, log):
    self.trips = trips

    if hasattr(self, "source_trip"):
      if len(self.trips[self.source_trip]) < 2:
        raise ValueError("Invalid trip")
    else:
      self.source_trip = np.random.randint(len(trips))
      while len(self.trips[self.source_trip]) < 2:
        self.source_trip = np.random.randint(len(trips))

    self.gift_to_move = self.gift_to_move if hasattr(self, "gift_to_move") else np.random.randint(len(self.trips[self.source_trip]))
    self.destination_trip = self.destination_trip if hasattr(self, "destination_trip") else self._get_valid_target_trip()
    self.destination_insertion_index = None
    super(MoveGiftToAnotherTripNeighbor, self).__init__(log)

  def _get_valid_target_trip(self):
    weight_of_gift = self.trips[self.source_trip][self.gift_to_move][utils.WEIGHT]
    for i in np.random.permutation(len(self.trips)):
      if i != self.source_trip and np.sum(self.trips[i][:, utils.WEIGHT]) + weight_of_gift <= utils.WEIGHT_LIMIT:
        return i

  def __str__(self):
    return "move-{}:{}-to-{}:{}".format(self.source_trip, self.gift_to_move,
        self.destination_trip, self.destination_insertion_index)

  @property
  @memoize
  def cost_delta(self):
    source = self.trips[self.source_trip]
    gift = source[self.gift_to_move]

    self.destination_insertion_index, cost_to_insert = self._find_best_insertion_index(
        self.trips[self.destination_trip], gift)

    cost_to_remove = self._cost_to_remove_gift(source, self.gift_to_move)

    total_cost = cost_to_insert + cost_to_remove

    return total_cost

  def apply(self):
    # self.log.debug("Applying {}".format(self))

    source = self.trips[self.source_trip]
    destination = self.trips[self.destination_trip]

    if self.VERIFY_COST_DELTA:
      old = utils.weighted_trip_length(source[:, utils.LOCATION], source[:, utils.WEIGHT]) + \
          utils.weighted_trip_length(destination[:, utils.LOCATION], destination[:, utils.WEIGHT])

    gift = source[self.gift_to_move]
    gift[utils.TRIP] = self.destination_trip+1

    destination = np.insert(destination, self.destination_insertion_index, gift, axis=0)
    self.trips[self.destination_trip] = destination

    source = np.delete(source, self.gift_to_move, axis=0)
    self.trips[self.source_trip] = source

    if self.VERIFY_COST_DELTA:
      new = utils.weighted_trip_length(source[:, utils.LOCATION], source[:, utils.WEIGHT]) + \
          utils.weighted_trip_length(destination[:, utils.LOCATION], destination[:, utils.WEIGHT])
      utils.verify_costs_are_equal(self.cost_delta, new-old)


class MoveGiftToLightestTripNeighbor(MoveGiftToAnotherTripNeighbor):
  def __init__(self, trips, log):
    self.destination_trip = self._get_lightest_target_trip(trips)

    self.source_trip = np.random.randint(len(trips))
    while len(self.trips[self.source_trip]) < 2:
      self.source_trip = np.random.randint(len(trips))
    self.gift_to_move = np.random.randint(len(trips[self.source_trip]))
    while self.source_trip == self.destination_trip or  trips[self.destination_trip][:, utils.WEIGHT].sum() + trips[self.source_trip][self.gift_to_move][utils.WEIGHT] > utils.WEIGHT_LIMIT:
      self.source_trip = np.random.randint(len(trips))
      self.gift_to_move = np.random.randint(len(trips[self.source_trip]))
    super(MoveGiftToLightestTripNeighbor, self).__init__(trips, log)

  def __str__(self):
    return "move-{}:{}-to-lightest-{}:{}".format(self.source_trip, self.gift_to_move,
        self.destination_trip, self.destination_insertion_index)

  def _get_lightest_target_trip(self, trips):
    # can be invalid!
    weights = [np.sum(trip[:, utils.WEIGHT]) for trip in trips]
    return weights.index(np.min(weights))


class SwapGiftsAcrossTripsNeighbor(Neighbor):
  def __init__(self, trips, log):
    # when trips/gifts to swap aren't specified, select them randomly
    self.trips = trips
    self.first_trip = np.random.randint(len(trips))
    while len(self.trips[self.first_trip]) < 3:
      self.first_trip = np.random.randint(len(trips))
    self.second_trip = np.random.randint(len(trips))
    while len(self.trips[self.second_trip]) < 3 or self.first_trip == self.second_trip:
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

    if self.VERIFY_COST_DELTA:
      old = utils.weighted_trip_length(first_trip[:, utils.LOCATION], first_trip[:, utils.WEIGHT]) + \
          utils.weighted_trip_length(second_trip[:, utils.LOCATION], second_trip[:, utils.WEIGHT])

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

    if self.VERIFY_COST_DELTA:
      new = utils.weighted_trip_length(first_trip[:, utils.LOCATION], first_trip[:, utils.WEIGHT]) + \
          utils.weighted_trip_length(second_trip[:, utils.LOCATION], second_trip[:, utils.WEIGHT])
      utils.verify_costs_are_equal(self.cost_delta, new-old)

