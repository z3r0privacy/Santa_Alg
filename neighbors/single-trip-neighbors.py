#!/usr/bin/env python

import numpy as np

import pandas as pd
import utils
from neighbor import Neighbor
from utils import memoize

# TODO: More single-trip neighbors like 3-opt, moving 2 adjacent nodes within trip, etc.

class SwapGiftsInTripNeighbor(Neighbor):
  def __init__(self, trip, first_gift=None, second_gift=None, log=None):
    # when gifts to swap aren't specified, select them randomly
    self.trip = trip
    self.first_gift = first_gift or np.random.randint(len(trip))
    self.second_gift = second_gift or np.random.randint(len(trip))
    while self.first_gift == self.second_gift:
      self.second_gift = np.random.randint(len(trip))
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

    # new_self.trip = self.trip[:]
    # old = utils.weighted_self.trip_length(pd.DataFrame(new_self.trip)[[utils.LAT, utils.LON]], list(pd.DataFrame(new_self.trip)[utils.WEIGHT]))
    # temp = new_self.trip[i]
    # new_self.trip[i] = new_self.trip[j]
    # new_self.trip[j] = temp
    # new = utils.weighted_self.trip_length(pd.DataFrame(new_self.trip)[[utils.LAT, utils.LON]], list(pd.DataFrame(new_self.trip)[utils.WEIGHT]))
    # return new - old

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
    self.trip[[self.first_gift, self.second_gift]] = self.trip[[self.second_gift, self.first_gift]]

