#!/usr/bin/env python

import numpy as np

import pandas as pd
import utils
from neighbor import Neighbor
from utils import memoize


class SwapRandomGiftsInTripNeighbor(Neighbor):
  def __init__(self, trips, log):
    # don't overwrite existing trip
    if not hasattr(self, "trip"):
      self.trip = trips[np.random.randint(len(trips))]
      while len(self.trip) < 2:
        self.trip = trips[np.random.randint(len(trips))]

    # don't overwrite existing gifts
    if not hasattr(self, "first_gift"):
      self.first_gift = np.random.randint(len(self.trip))

    if not hasattr(self, "second_gift"):
      self.second_gift = np.random.randint(len(self.trip))
      while self.first_gift == self.second_gift:
        self.second_gift = np.random.randint(len(self.trip))
    super(SwapRandomGiftsInTripNeighbor, self).__init__(log)

  def __str__(self):
    return "{}-random-swap-{}-{}".format(int(self.trip[0][1]), self.first_gift, self.second_gift)

  def _get_cost_of_swapping_adjacent(self, a, b, c, d, cumulative_weight_at_a, weight_at_b, weight_at_c):
    old_cost = utils.distance(a, b) * cumulative_weight_at_a + \
        utils.distance(b, c) * (cumulative_weight_at_a - weight_at_b) + \
        utils.distance(c, d) * (cumulative_weight_at_a - weight_at_b - weight_at_c)
    new_cost = utils.distance(a, c) * cumulative_weight_at_a + \
        utils.distance(c, b) * (cumulative_weight_at_a - weight_at_c) + \
        utils.distance(b, d) * (cumulative_weight_at_a - weight_at_c - weight_at_b)
    return new_cost - old_cost

  def _calculate_cost_of_swapping_items(self, first, second):
    i = min(first, second)
    j = max(first, second)

    # set up weights
    weight_diff = self.trip[i][utils.WEIGHT] - self.trip[j][utils.WEIGHT]
    cum_weight_before_i = np.sum(self.trip[i:][:, utils.WEIGHT]) + utils.SLEIGH_WEIGHT
    cum_weight_before_j = np.sum(self.trip[j:][:, utils.WEIGHT]) + utils.SLEIGH_WEIGHT
    weight_i = self.trip[i][utils.WEIGHT]
    weight_j = self.trip[j][utils.WEIGHT]

    # set up locations
    before_i = tuple(self.trip[i-1][[utils.LAT, utils.LON]]) if i > 0 else utils.NORTH_POLE
    before_j = tuple(self.trip[j-1][[utils.LAT, utils.LON]]) if j > 0 else utils.NORTH_POLE
    at_i = tuple(self.trip[i][[utils.LAT, utils.LON]])
    at_j = tuple(self.trip[j][[utils.LAT, utils.LON]])
    after_i = tuple(self.trip[i+1][[utils.LAT, utils.LON]]) if i < len(self.trip)-1 else utils.NORTH_POLE
    after_j = tuple(self.trip[j+1][[utils.LAT, utils.LON]]) if j < len(self.trip)-1 else utils.NORTH_POLE

    if i+1 == j:
      # swap adjacent locations is simplified
      improvement = self._get_cost_of_swapping_adjacent(before_i, at_i, at_j, after_j,
          cum_weight_before_i, weight_i, weight_j)
    else:
      # cost of the old segments around i/j
      old_i = Neighbor.get_cost_of_tour_of_three(before_i, at_i, after_i, cum_weight_before_i, weight_i)
      old_j = Neighbor.get_cost_of_tour_of_three(before_j, at_j, after_j, cum_weight_before_j, weight_j)

      # cost of the new segments around i/j
      new_j = Neighbor.get_cost_of_tour_of_three(before_i, at_j, after_i, cum_weight_before_i, weight_j)
      new_i = Neighbor.get_cost_of_tour_of_three(before_j, at_i, after_j, cum_weight_before_j + weight_diff, weight_i)

      # cost difference from weight between i and j (sub-self.trip between i+1..j-1)
      distance = 0
      for k in range(i+1, j-1):
        distance += utils.distance(
            tuple(self.trip[k][[utils.LAT, utils.LON]]),
            tuple(self.trip[k+1][[utils.LAT, utils.LON]])
            )
      diff = distance * weight_diff
      improvement = new_j + new_i - old_j - old_i + diff

    return improvement


  @property
  @memoize
  def cost_delta(self):
    return self._calculate_cost_of_swapping_items(self.first_gift, self.second_gift)

  def apply(self):
    # self.log.debug("Applying {}".format(self))

    if self.VERIFY_COST_DELTA:
      old = utils.weighted_trip_length(self.trip[:, utils.LOCATION], self.trip[:, utils.WEIGHT])

    self.trip[[self.first_gift, self.second_gift]] = self.trip[[self.second_gift, self.first_gift]]

    if self.VERIFY_COST_DELTA:
      new = utils.weighted_trip_length(self.trip[:, utils.LOCATION], self.trip[:, utils.WEIGHT])
      utils.verify_costs_are_equal(self.cost_delta, new-old)


class OptimalSwapInRandomTripNeighbor(SwapRandomGiftsInTripNeighbor):
  def __init__(self, trips, log, trip=None, first_gift=None):
    if trip is not None:
      # select random trip with at least 2 gifts
      self.trip = trips[np.random.randint(len(trips))]
      while len(self.trip) < 2:
        self.trip = trips[np.random.randint(len(trips))]

    if first_gift is not None:
      # select the first gift to swap randomly
      self.first_gift = np.random.randint(len(self.trip))

    # don't assign second gift yet
    self.second_gift = -1

    super(OptimalSwapInRandomTripNeighbor, self).__init__(trips, log)

  def __str__(self):
    return "{}-optimal-swap-{}-{}".format(int(self.trip[0][1]), self.first_gift, self.second_gift)

  @property
  @memoize
  def cost_delta(self):
    minimum_cost = np.finfo(np.float64).max
    self.second_gift = None

    for i in range(len(self.trip)):
      if i == self.first_gift:
        # don't swap with self
        continue
      current_cost = self._calculate_cost_of_swapping_items(self.first_gift, i)
      if current_cost < minimum_cost:
        minimum_cost = current_cost
        self.second_gift = i

    return minimum_cost

