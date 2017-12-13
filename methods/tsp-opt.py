#!/usr/bin/env python

import numpy as np

import pandas as pd
import utils
from method import Method
from neighbors import OptimalSwapInRandomTripNeighbor


class RandomTspOptimizeTripMethod(Method):
  def __init__(self, gifts, log):
    super(RandomTspOptimizeTripMethod, self).__init__(gifts, log)

  @property
  def name(self):
    return "randomtriptsp"

  def run(self, args):
    """
    Idea: Load existing valid solution and improve it by improving the order of gift distribution
    within a single trip. For each trip, select x elements and swap it with the other element that
    yields the largest decrease in cost.
    """
    all_trips = self._load_trips_from_file(args)
    if all_trips is None:
      return
    swaps_per_trip = 10

    # split all trips into separate trips
    trips = [all_trips[all_trips.TripId == t].values for t in all_trips.TripId.unique()]

    # process each trip separately
    for i, trip in enumerate(trips):
      if i % 100 == 0:
        self.log.info("Optimizing trip {}...".format(i))

      # for each trip, swap x random items
      total_improvement = 0
      swaps = 0
      for _ in range(swaps_per_trip):
        neighbor = OptimalSwapInRandomTripNeighbor(trips, self.log, trip=trip)
        if neighbor.cost_delta < 0:
          total_improvement += neighbor.cost_delta
          swaps += 1
          neighbor.apply()

      self.log.debug("Swapped {} locations with a total improvement of {}".format(swaps, total_improvement))

      # done processing the trip, write back the permutated trip
      trips[i] = trip

    # extract gift/trip mapping
    combined_trips = np.concatenate(trips)[:, [utils.GIFT, utils.TRIP]]
    self.trips = pd.DataFrame(combined_trips, columns=["GiftId", "TripId"])


class ThoroughTspOptimizeTripMethod(Method):
  def __init__(self, gifts, log):
    super(ThoroughTspOptimizeTripMethod, self).__init__(gifts, log)

  @property
  def name(self):
    return "thoroughtriptsp"

  def run(self, args):
    """
    Idea: Load existing valid solution and improve it by improving the order of gift distribution
    within a single trip. Until at least half the elements weren't moved, improve each trip by
    swapping an element with one that provides the largest decrease in cost.
    """
    all_trips = self._load_trips_from_file(args)
    if all_trips is None:
      return

    # split all trips into separate trips
    trips = [all_trips[all_trips.TripId == t].values for t in all_trips.TripId.unique()]

    # process each trip separately
    for i, trip in enumerate(trips):
      if i % 100 == 0:
        self.log.info("Optimizing trip {}...".format(i))

      # try to improve until swapping half the gifts failed to improve the solution
      swaps = 0
      total_improvement = 0
      no_improvement_count = 0
      abort_after_x_without_improvement = len(trip) / 2
      # try gifts in random order
      indexes = np.random.permutation(range(len(trip)))
      for index in indexes:
        neighbor = OptimalSwapInRandomTripNeighbor(trips, self.log, trip=trip, first_gift=index)
        if neighbor.cost_delta < 0:
          total_improvement += neighbor.cost_delta
          swaps += 1
          neighbor.apply()
        else:
          no_improvement_count += 1
          if no_improvement_count > abort_after_x_without_improvement: # TODO: try considering successful swap count
            break

      self.log.debug("Checked {:>3d} and swapped {:>2d} locations with a total improvement of {:.3}M".format(
        swaps + no_improvement_count, swaps, total_improvement / 1e6))

      # done processing the trip, write back the permutated trip
      trips[i] = trip

    # extract gift/trip mapping
    combined_trips = np.concatenate(trips)[:, [utils.GIFT, utils.TRIP]]
    self.trips = pd.DataFrame(combined_trips, columns=["GiftId", "TripId"])

