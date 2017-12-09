#!/usr/bin/env python

import glob

import numpy as np

import pandas as pd
import utils
from method import Method


class TspOptimizeTripMethod(Method):
  @property
  def name(self):
    return "triptsp"

  def get_cost_of_tour_of_three(self, a, b, c, cumulative_weight_at_a, weight_at_b):
    return utils.distance(a, b) * cumulative_weight_at_a + \
        utils.distance(b, c) * (cumulative_weight_at_a - weight_at_b)

  def get_cost_of_swapping_adjacent(self, a, b, c, d, cumulative_weight_at_a, weight_at_b, weight_at_c):
    old_cost = utils.distance(a, b) * cumulative_weight_at_a + \
        utils.distance(b, c) * (cumulative_weight_at_a - weight_at_b) + \
        utils.distance(c, d) * (cumulative_weight_at_a - weight_at_b - weight_at_c)
    new_cost = utils.distance(a, c) * cumulative_weight_at_a + \
        utils.distance(c, b) * (cumulative_weight_at_a - weight_at_c) + \
        utils.distance(b, d) * (cumulative_weight_at_a - weight_at_c - weight_at_b)
    return new_cost - old_cost

  def get_improvement_of_swapping(self, trip, first, second):
    i = min(first, second)
    j = max(first, second)

    lat = 2
    lon = 3
    weight = 4

    # new_trip = trip[:]
    # old = utils.weighted_trip_length(pd.DataFrame(new_trip)[[lat, lon]], list(pd.DataFrame(new_trip)[weight]))
    # temp = new_trip[i]
    # new_trip[i] = new_trip[j]
    # new_trip[j] = temp
    # new = utils.weighted_trip_length(pd.DataFrame(new_trip)[[lat, lon]], list(pd.DataFrame(new_trip)[weight]))
    # return new - old

    # set up weights
    weight_diff = trip[i][weight] - trip[j][weight]
    cum_weight_before_i = np.sum(trip[i:][:, weight]) + utils.SLEIGH_WEIGHT
    cum_weight_before_j = np.sum(trip[j:][:, weight]) + utils.SLEIGH_WEIGHT
    weight_i = trip[i][weight]
    weight_j = trip[j][weight]

    # set up locations
    before_i = tuple(trip[i-1][[lat, lon]]) if i > 0 else utils.NORTH_POLE
    before_j = tuple(trip[j-1][[lat, lon]]) if j > 0 else utils.NORTH_POLE
    at_i = tuple(trip[i][[lat, lon]])
    at_j = tuple(trip[j][[lat, lon]])
    after_i = tuple(trip[i+1][[lat, lon]]) if i < len(trip)-1 else utils.NORTH_POLE
    after_j = tuple(trip[j+1][[lat, lon]]) if j < len(trip)-1 else utils.NORTH_POLE

    if i+1 == j:
      # swap adjacent locations is simplified
      improvement = self.get_cost_of_swapping_adjacent(before_i, at_i, at_j, after_j,
          cum_weight_before_i, weight_i, weight_j)
    else:
      # cost of the old segments around i/j
      old_i = self.get_cost_of_tour_of_three(before_i, at_i, after_i, cum_weight_before_i, weight_i)
      old_j = self.get_cost_of_tour_of_three(before_j, at_j, after_j, cum_weight_before_j, weight_j)

      # cost of the new segments around i/j
      new_j = self.get_cost_of_tour_of_three(before_i, at_j, after_i, cum_weight_before_i, weight_j)
      new_i = self.get_cost_of_tour_of_three(before_j, at_i, after_j, cum_weight_before_j + weight_diff, weight_i)

      # cost difference from weight between i and j (sub-trip between i+1..j-1)
      distance = 0
      for k in range(i+1, j-1):
        distance += utils.distance(
            tuple(trip[k][[lat, lon]]),
            tuple(trip[k+1][[lat, lon]])
            )
      diff = distance * weight_diff
      improvement = new_j + new_i - old_j - old_i + diff

    return improvement

  def run(self, args):
    """
    Idea: Load existing valid solution and improve it by improving the order of gift distribution
    within a single trip. Change order by randomly selecting x locations and swapping them with the
    one that provides the largest decrease in cost.
    """
    matches = glob.glob("data/*{}*".format(args.from_file))
    if not matches:
      self.log.warning("No matching file found, aborting!")
      return

    self.log.info("Using file {} from {} matching files ({})".format(matches[0], len(matches), matches))
    all_trips = pd.read_csv(matches[0]).merge(self.gifts, on="GiftId")
    swaps_per_trip = 10

    # split all trips into separate trips
    trips = [all_trips[all_trips.TripId == t].values for t in all_trips.TripId.unique()]

    # process each trip separately
    for i, trip in enumerate(trips):
      if i % 100 == 0:
        self.log.info("Optimizing trip {}...".format(i))

      # for each trip, swap x random items
      total_improvement = 0
      for foobar in range(swaps_per_trip):
        index = np.random.randint(trip.shape[0])
        swap_with_index = 0
        improvement = self.get_improvement_of_swapping(trip, index, swap_with_index)

        for j in range(trip.shape[0]):
          if j == index:
            continue
          current_improvement = self.get_improvement_of_swapping(trip, index, j)
          if current_improvement < improvement:
            swap_with_index = j
            improvement = current_improvement

        if improvement >= 0:
          self.log.debug("No improvement found from swapping {}, trying another...".format(index))
          foobar -= 1
          continue
        total_improvement += improvement
        trip[[index, swap_with_index]] = trip[[swap_with_index, index]]

      self.log.debug("Swapped {} locations with a total improvement of {}".format(swaps_per_trip, total_improvement))

      # done processing the trip, write back the permutated trip
      trips[i] = trip

    # extract gift/trip mapping
    combined_trips = np.concatenate(trips)[:, [0, 1]]
    self.trips = pd.DataFrame(combined_trips, columns=["GiftId", "TripId"])

