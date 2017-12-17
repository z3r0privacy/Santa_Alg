#!/usr/bin/env python

import copy

import numpy as np
import pandas as pd

import utils
from neighbor import Neighbor


class MoveGiftToAnotherTripNeighbor(Neighbor):
  def __init__(self, trips):
    self.trips = trips
    if not hasattr(self, "source_trip"):
      self.source_trip = None
    if not hasattr(self, "gift_to_move"):
      self.gift_to_move = None
    self.destination_trip = None
    self.destination_insertion_index = None
    self.cost_to_insert_in_destination = None
    super(MoveGiftToAnotherTripNeighbor, self).__init__()

  def _get_valid_target_trip(self):
    weight_of_gift = self.trips[self.source_trip][self.gift_to_move][utils.WEIGHT]
    for i in np.random.permutation(len(self.trips)):
      if i != self.source_trip and np.sum(self.trips[i][:, utils.WEIGHT]) + weight_of_gift <= utils.WEIGHT_LIMIT:
        return i

  def __str__(self):
    return "move-{}:{}-to-{}:{}: {:.5f}M".format(self.source_trip, self.gift_to_move,
        self.destination_trip, self.destination_insertion_index, self.cost_delta() / 1e6)

  def cost_delta(self):
    if self.cost is not None:
      return self.cost

    if self.source_trip is None:
      self.source_trip = np.random.randint(len(self.trips))
      while len(self.trips[self.source_trip]) < 2:
        self.source_trip = np.random.randint(len(self.trips))

    source = self.trips[self.source_trip]

    if self.gift_to_move is not None or self.destination_trip is not None or self.destination_insertion_index is not None or self.cost_to_insert_in_destination is not None:
      # we should have *all* of these set
      return self.cost_to_insert_in_destination + self._cost_to_remove_gift(source, self.gift_to_move)

    self.gift_to_move = np.random.randint(len(source))
    self.destination_trip = self._get_valid_target_trip()

    gift = source[self.gift_to_move]

    self.destination_insertion_index, cost_to_insert = Neighbor.find_best_insertion_index(
        self.trips[self.destination_trip], gift)

    cost_to_remove = self._cost_to_remove_gift(source, self.gift_to_move)

    self.cost = cost_to_insert + cost_to_remove
    return self.cost

  def apply(self):
    # self.log.debug("Applying {}".format(self))

    source = self.trips[self.source_trip]
    destination = self.trips[self.destination_trip]

    if self.VERIFY_COST_DELTA:
      old = utils.weighted_trip_length(source[:, utils.LOCATION], source[:, utils.WEIGHT]) + \
          utils.weighted_trip_length(destination[:, utils.LOCATION], destination[:, utils.WEIGHT])

    gift = source[self.gift_to_move] # NOTE: This apparently can be index-out-of-bounds!
    gift[utils.TRIP] = destination[0, utils.TRIP]

    destination = np.insert(destination, self.destination_insertion_index, gift, axis=0)
    self.trips[self.destination_trip] = destination

    source = np.delete(source, self.gift_to_move, axis=0)
    self.trips[self.source_trip] = source

    if self.VERIFY_COST_DELTA:
      new = utils.weighted_trip_length(source[:, utils.LOCATION], source[:, utils.WEIGHT]) + \
          utils.weighted_trip_length(destination[:, utils.LOCATION], destination[:, utils.WEIGHT])
      utils.verify_costs_are_equal(self.cost_delta(), new-old)


class MoveGiftToLightestTripNeighbor(MoveGiftToAnotherTripNeighbor):
  def __init__(self, trips):
    self.destination_trip = self._get_lightest_target_trip(trips)

    self.source_trip = np.random.randint(len(trips))
    while len(self.trips[self.source_trip]) < 2:
      self.source_trip = np.random.randint(len(trips))
    self.gift_to_move = np.random.randint(len(trips[self.source_trip]))
    while self.source_trip == self.destination_trip or  trips[self.destination_trip][:, utils.WEIGHT].sum() + trips[self.source_trip][self.gift_to_move][utils.WEIGHT] > utils.WEIGHT_LIMIT:
      self.source_trip = np.random.randint(len(trips))
      self.gift_to_move = np.random.randint(len(trips[self.source_trip]))
    super(MoveGiftToLightestTripNeighbor, self).__init__(trips)

  def __str__(self):
    return "move-{}:{}-to-lightest-{}:{}".format(self.source_trip, self.gift_to_move,
        self.destination_trip, self.destination_insertion_index)

  def _get_lightest_target_trip(self, trips):
    # can be invalid!
    weights = [np.sum(trip[:, utils.WEIGHT]) for trip in trips]
    return weights.index(np.min(weights))


class MoveGiftToOptimalTripNeighbor(MoveGiftToAnotherTripNeighbor):
  def __init__(self, trips, trip=None, gift_to_move=None):
    if trip is None:
      self.source_trip = np.random.randint(len(trips))
      while len(trips[self.source_trip]) < 2:
        self.source_trip = np.random.randint(len(trips))
    else:
      self.source_trip = trip
    self.gift_to_move = np.random.randint(len(trips[self.source_trip])) if gift_to_move is None else gift_to_move
    super(MoveGiftToOptimalTripNeighbor, self).__init__(trips)

  def __str__(self):
    return "move-{}:{}-to-optimal-{}:{}".format(self.source_trip, self.gift_to_move,
        self.destination_trip, self.destination_insertion_index)

  def find_close_trips(self, gift, trip_index_to_skip):
    gift_longitude = gift[utils.LON]
    gift_weight = gift[utils.WEIGHT]
    tolerance = 1
    while True:
      candidates = []
      close_candidates = []
      for i, trip in enumerate(self.trips):
        if i == trip_index_to_skip or trip[:, utils.WEIGHT].sum() + gift_weight > utils.WEIGHT_LIMIT:
          # avoid full candidates and moving to same trip
          continue
        min_lon = trip[:, utils.LON].min()
        max_lon = trip[:, utils.LON].max()
        if gift_longitude > min_lon and gift_longitude < max_lon:
          candidates.append(i)
        elif gift_longitude > min_lon - tolerance and gift_longitude < max_lon + tolerance:
          close_candidates.append(i)
      if candidates:
        return candidates
      if close_candidates:
        return close_candidates
      tolerance += 1

  def cost_delta(self):
    if self.cost is not None:
      return self.cost

    gift = self.trips[self.source_trip][self.gift_to_move]

    # find candidates for optimal destination trip
    # trips are good candidates if inserting the gift doesn't add a (big) detour
    candidate_trips = self.find_close_trips(gift, self.source_trip)

    best_candidate = None
    best_index_in_candidate = None
    minimum_cost = np.finfo(np.float64).max

    # try inserting into each
    for candidate in candidate_trips:
      destination_index, cost = self.find_best_insertion_index(self.trips[candidate], gift)
      if cost < minimum_cost:
        minimum_cost = cost
        best_candidate = candidate
        best_index_in_candidate = destination_index

    self.destination_trip = best_candidate
    self.destination_insertion_index = best_index_in_candidate
    self.cost_to_insert_in_destination = minimum_cost

    self.cost = super(MoveGiftToOptimalTripNeighbor, self).cost_delta()
    return self.cost


class SwapGiftsAcrossTripsNeighbor(Neighbor):
  def __init__(self, trips):
    self.trips = trips
    self.first_trip = np.random.randint(len(trips))
    while len(self.trips[self.first_trip]) < 3:
      self.first_trip = np.random.randint(len(trips))

    self.second_trip = None
    self.first_gift = None
    self.second_gift = None
    self.first_trip_insertion_index = None
    self.second_trip_insertion_index = None
    super(SwapGiftsAcrossTripsNeighbor, self).__init__()

  def __str__(self):
    return "swap-{}:{}-{}:{}".format(self.first_trip, self.first_gift, self.second_trip, self.second_gift)

  def _get_valid_swapee(self):
    weight_of_first_gift = self.trips[self.first_trip][self.first_gift][utils.WEIGHT]
    first_weight = np.sum(self.trips[self.first_trip][:, utils.WEIGHT])
    second_weight = np.sum(self.trips[self.second_trip][:, utils.WEIGHT])

    for gift in np.random.permutation(len(self.trips[self.second_trip])):
      weight_of_second_gift = self.trips[self.second_trip][gift][utils.WEIGHT]
      if (first_weight - weight_of_first_gift + weight_of_second_gift <= utils.WEIGHT_LIMIT and
          second_weight + weight_of_first_gift - weight_of_second_gift <= utils.WEIGHT_LIMIT):
        return gift
    return None

  def cost_delta(self):
    if self.cost is not None:
      return self.cost

    # find second trip to exchange gifts with and valid gifts to swap
    first_gifts_to_try = np.random.permutation(len(self.trips[self.first_trip]))
    for fg in first_gifts_to_try:
      self.second_trip = np.random.randint(len(self.trips))
      while len(self.trips[self.second_trip]) < 3 or self.first_trip == self.second_trip:
        self.second_trip = np.random.randint(len(self.trips))
      self.first_gift = fg
      self.second_gift = self._get_valid_swapee()
      if self.second_gift is not None:
        break

    # find insertion indexes with minimum cost
    self.first_trip_insertion_index, cost_to_insert_first = Neighbor.find_best_insertion_index(
        self.trips[self.first_trip], self.trips[self.second_trip][self.second_gift], index_to_be_removed=self.first_gift)
    self.second_trip_insertion_index, cost_to_insert_second = Neighbor.find_best_insertion_index(
        self.trips[self.second_trip], self.trips[self.first_trip][self.first_gift], index_to_be_removed=self.second_gift)

    # update temporary trips with new insertion to accurately calculate the cost of deletion
    temporary_first_trip = self.trips[self.first_trip]
    temporary_first_trip = np.insert(temporary_first_trip, self.first_trip_insertion_index, self.trips[self.second_trip][self.second_gift], axis=0)
    temporary_second_trip = self.trips[self.second_trip]
    temporary_second_trip = np.insert(temporary_second_trip, self.second_trip_insertion_index, self.trips[self.first_trip][self.first_gift], axis=0)

    # calculate (negative) cost of deletion
    cost_to_remove_first = self._cost_to_remove_gift(temporary_first_trip, self.first_gift if self.first_gift < self.first_trip_insertion_index else self.first_gift + 1)
    cost_to_remove_second = self._cost_to_remove_gift(temporary_second_trip, self.second_gift if self.second_gift < self.second_trip_insertion_index else self.second_gift + 1)

    self.cost = cost_to_insert_first + cost_to_insert_second + cost_to_remove_first + cost_to_remove_second

    return self.cost

  def apply(self):
    # self.log.debug("Applying {}".format(self))

    first_trip = self.trips[self.first_trip]
    second_trip = self.trips[self.second_trip]

    if self.VERIFY_COST_DELTA:
      old = utils.weighted_trip_length(first_trip[:, utils.LOCATION], first_trip[:, utils.WEIGHT]) + \
          utils.weighted_trip_length(second_trip[:, utils.LOCATION], second_trip[:, utils.WEIGHT])

    # extract insertees now (before they're removed) and update their trip assignment
    first_gift_row = first_trip[self.first_gift]
    trip_id_for_second_gift = first_trip[0, utils.TRIP]
    first_gift_row[utils.TRIP] = second_trip[0, utils.TRIP]
    second_gift_row = second_trip[self.second_gift]
    second_gift_row[utils.TRIP] = trip_id_for_second_gift

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
      utils.verify_costs_are_equal(self.cost_delta(), new-old)


class OptimalMergeTripIntoAdjacentNeighbor(Neighbor):
  def __init__(self, trips, trip=None):
    self.trips = trips
    self.trip_to_merge = trip
    self.trips_with_applied_merge = None
    self.modified_trips = None
    super(OptimalMergeTripIntoAdjacentNeighbor, self).__init__()

  def __str__(self):
    return "merge-{}-optimally".format(self.trip_to_merge)

  def _find_trip_to_merge(self):
    # TODO: Find reasonable heuristics
    weights = [np.sum(trip[:, utils.WEIGHT]) for trip in self.trips]
    maximum_weight = min(500, np.median(weights), np.mean(weights))
    lengths = [len(trip[:, utils.WEIGHT]) for trip in self.trips]
    maximum_trip_length = min(50, np.median(lengths), np.mean(lengths))

    trips_to_check = np.random.permutation(len(self.trips))

    for trip_index in trips_to_check:
      trip = self.trips[trip_index]
      trip_weight = np.sum(trip[:, utils.WEIGHT])
      if trip_weight < maximum_weight and len(trip) < maximum_trip_length:
        return trip_index

  def cost_delta(self):
    if self.cost is not None:
      return self.cost

    if self.trip_to_merge is None:
      self.trip_to_merge = self._find_trip_to_merge()
    if self.trip_to_merge is None:
      return 0

    trip = self.trips[self.trip_to_merge]

    # sort gifts by descending weight
    sorting_indices = trip[:, utils.WEIGHT].argsort()[::-1]
    sorted_gifts = trip[sorting_indices]

    # move gifts by size to optimal other trips
    self.trips_with_applied_merge = copy.deepcopy(self.trips)
    self.cost = 0
    self.modified_trips = []
    for gift in sorted_gifts:
      # find the index of the current gift in the current trip that is being merged
      gift_index = -1
      for i, g in enumerate(self.trips_with_applied_merge[self.trip_to_merge]):
        if g[utils.GIFT] == gift[utils.GIFT]:
          gift_index = i
          break
      # move gift
      neighbor = MoveGiftToOptimalTripNeighbor(self.trips_with_applied_merge, self.trip_to_merge, gift_index)
      self.cost += neighbor.cost_delta()
      neighbor.apply()
      if not neighbor.destination_trip in self.modified_trips:
        self.modified_trips.append(neighbor.destination_trip)

    return self.cost


  def apply(self):
    if self.trip_to_merge is None:
      Neighbor.log.warning("Not applying trip merge because no valid merge was found")
      return

    # self.log.debug("Applying {}".format(self))

    trip = self.trips[self.trip_to_merge]

    if self.VERIFY_COST_DELTA:
      old = utils.weighted_trip_length(trip[:, utils.LOCATION], trip[:, utils.WEIGHT])
      for trip_index in self.modified_trips:
        old += utils.weighted_trip_length(self.trips[trip_index][:, utils.LOCATION], self.trips[trip_index][:, utils.WEIGHT])

    self.trips.clear()
    self.trips.extend(self.trips_with_applied_merge)

    if self.VERIFY_COST_DELTA:
      new = 0
      for trip_index in self.modified_trips:
        new += utils.weighted_trip_length(self.trips[trip_index][:, utils.LOCATION], self.trips[trip_index][:, utils.WEIGHT])
      utils.verify_costs_are_equal(self.cost_delta(), new-old)

    # only delete the row afterwards to not mess up the indexes for the cost calculation
    del self.trips[self.trip_to_merge]

