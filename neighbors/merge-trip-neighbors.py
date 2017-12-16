#!/usr/bin/env python

import numpy as np

import pandas as pd
import utils
from neighbor import Neighbor


class MergeTripIntoOthersNeighbor(Neighbor):
  def __init__(self, trips):
    self.trips = trips
    self.trip_to_merge = None
    self.trip_assignments_for_gifts = None
    self.gift_insertions = None
    super(MergeTripIntoOthersNeighbor, self).__init__()

  def __str__(self):
    return "merge-{}-into-others".format(self.trip_to_merge)

  def _find_trip_to_merge(self):
    weights = [np.sum(trip[:, utils.WEIGHT]) for trip in self.trips]
    median_weight = np.median(weights)

    trips_to_check = np.random.permutation(len(self.trips))

    # check all trips (in random order) to see whether they may be merged into the other trips
    for trip_index in trips_to_check:
      trip = self.trips[trip_index]
      trip_weight = np.sum(trip[:, utils.WEIGHT])
      if trip_weight > median_weight:
        # large trips are unlikely to be merged successfully
        continue

      # sort gifts by descending weight
      sorting_indices = trip[:, utils.WEIGHT].argsort()[::-1]
      sorted_gifts = trip[sorting_indices]

      # try to assign all gifts of the current trip to other trips
      gift_assignment = {}
      trip_failed = False
      for gift in sorted_gifts:
        # find a new trip that can accommodate the gift - we only try trips that haven't been modified yet
        # that 1. is more balanced and 2. makes it possible to easily calculate costs
        host_found = False
        shuffled_trips = [(i, t) for i, t in enumerate(self.trips)]
        np.random.shuffle(shuffled_trips)
        for i, host_trip in shuffled_trips:
          # print(i, host_trip)
          if i in gift_assignment.keys() or i == trip_index:
            # we don't want to insert into trips that already receive a new item or into the trip we're trying to merge
            continue
          if gift[utils.WEIGHT] + host_trip[:, utils.WEIGHT].sum() < utils.WEIGHT_LIMIT:
            gift_assignment[i] = gift
            host_found = True
            break
        if not host_found:
          trip_failed = True
          break

      if not trip_failed:
        return trip_index, gift_assignment

      Neighbor.log.warning("Failed to spread the gifts of a trip with weight {} (median: {})".format(trip_weight, median_weight))
    return None, None

  def cost_delta(self):
    if self.cost is not None:
      return self.cost

    self.trip_to_merge, self.trip_assignments_for_gifts = self._find_trip_to_merge()

    if self.trip_to_merge is None:
      return 0

    trip = self.trips[self.trip_to_merge]
    self.gift_insertions = []
    cost_of_insertions = 0

    for trip_index, gift in self.trip_assignments_for_gifts.items():
      index_in_trip, cost = Neighbor.find_best_insertion_index(self.trips[trip_index], gift)
      self.gift_insertions.append((gift, trip_index, index_in_trip))
      cost_of_insertions += cost

    self.cost = cost_of_insertions - utils.weighted_trip_length(trip[:, utils.LOCATION], trip[:, utils.WEIGHT])
    return self.cost

  def apply(self):
    if self.trip_to_merge is None:
      Neighbor.log.warning("Not applying trip merge because no valid merge was found")
      return

    # self.log.debug("Applying {}".format(self))

    trip = self.trips[self.trip_to_merge]

    if self.VERIFY_COST_DELTA:
      old = utils.weighted_trip_length(trip[:, utils.LOCATION], trip[:, utils.WEIGHT])
      for trip_index in self.trip_assignments_for_gifts.keys():
        old += utils.weighted_trip_length(self.trips[trip_index][:, utils.LOCATION], self.trips[trip_index][:, utils.WEIGHT])

    for gift, trip_index, index_in_trip in self.gift_insertions:
      gift[utils.TRIP] = self.trips[trip_index][0,1]
      self.trips[trip_index] = np.insert(self.trips[trip_index], index_in_trip, gift, axis=0)

    if self.VERIFY_COST_DELTA:
      new = 0
      for trip_index in self.trip_assignments_for_gifts.keys():
        new += utils.weighted_trip_length(self.trips[trip_index][:, utils.LOCATION], self.trips[trip_index][:, utils.WEIGHT])
      utils.verify_costs_are_equal(self.cost_delta(), new-old)

    # only delete the row afterwards to not mess up the indexes for the cost calculation
    del self.trips[self.trip_to_merge]

