#!/usr/bin/env python

import numpy as np

import pandas as pd
import utils
from neighbor import Neighbor
from utils import memoize


class SwapGiftsInTripNeighbor(Neighbor):
  def __init__(self, trips, log):
    # when gifts to swap aren't specified, select them randomly
    self.trip = trips[np.random.randint(len(trips))]
    while len(self.trip) < 2:
      self.trip = trips[np.random.randint(len(trips))]
    self.first_gift = np.random.randint(len(self.trip))
    self.second_gift = np.random.randint(len(self.trip))
    while self.first_gift == self.second_gift:
      self.second_gift = np.random.randint(len(self.trip))
    super(SwapGiftsInTripNeighbor, self).__init__(log)

  def __str__(self):
    return "{}-swap-{}-{}".format(int(self.trip[0][1]), self.first_gift, self.second_gift)

  def _get_cost_of_swapping_adjacent(self, a, b, c, d, cumulative_weight_at_a, weight_at_b, weight_at_c):
    old_cost = utils.distance(a, b) * cumulative_weight_at_a + \
        utils.distance(b, c) * (cumulative_weight_at_a - weight_at_b) + \
        utils.distance(c, d) * (cumulative_weight_at_a - weight_at_b - weight_at_c)
    new_cost = utils.distance(a, c) * cumulative_weight_at_a + \
        utils.distance(c, b) * (cumulative_weight_at_a - weight_at_c) + \
        utils.distance(b, d) * (cumulative_weight_at_a - weight_at_c - weight_at_b)
    return new_cost - old_cost

  @property
  @memoize
  def cost_delta(self):
    i = min(self.first_gift, self.second_gift)
    j = max(self.first_gift, self.second_gift)

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
      old_i = self.get_cost_of_tour_of_three(before_i, at_i, after_i, cum_weight_before_i, weight_i)
      old_j = self.get_cost_of_tour_of_three(before_j, at_j, after_j, cum_weight_before_j, weight_j)

      # cost of the new segments around i/j
      new_j = self.get_cost_of_tour_of_three(before_i, at_j, after_i, cum_weight_before_i, weight_j)
      new_i = self.get_cost_of_tour_of_three(before_j, at_i, after_j, cum_weight_before_j + weight_diff, weight_i)

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

  def apply(self):
    # self.log.debug("Applying {}".format(self))

    if self.VERIFY_COST_DELTA:
      old = utils.weighted_trip_length(self.trip[:, utils.LOCATION], self.trip[:, utils.WEIGHT])

    self.trip[[self.first_gift, self.second_gift]] = self.trip[[self.second_gift, self.first_gift]]

    if self.VERIFY_COST_DELTA:
      new = utils.weighted_trip_length(self.trip[:, utils.LOCATION], self.trip[:, utils.WEIGHT])
      utils.verify_costs_are_equal(self.cost_delta, new-old)

