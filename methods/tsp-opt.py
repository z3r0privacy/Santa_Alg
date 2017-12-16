#!/usr/bin/env python

import numpy as np
import pandas as pd
import utils
from method import Method
from neighbors import (OptimalMoveGiftInTripNeighbor,
                       OptimalSwapInRandomTripNeighbor)


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
      if i % 100 == 0 and i != 0:
        checkpoint_file = "checkpoints/{}_{}_{}.csv".format(args.evaluation_id, args.random_seed, i)
        self.log.info("{:>6}/{}: Creating checkpoint '{}'".format(i, len(trips), checkpoint_file))
        combined_trips = np.concatenate(trips)[:, [0, 1]]
        self.trips = pd.DataFrame(combined_trips, columns=["GiftId", "TripId"])
        self.write_trips(checkpoint_file)

      swaps = 0
      improvement = 0
      current_improvement = 0
      no_improvement_count = 0
      abort_after_x_without_improvement = int(len(trip) * 0.9)
      min_tries = 10
      min_improvement = -1e2
      tries = 0
      # try at least min_tries times and as long as there's some "considerable" improvement
      while current_improvement < min_improvement or min_tries > tries:
        current_improvement = 0
        current_swaps = 0
        current_no_improvement_count = 0
        tries += 1

        # try to improve in random order until swapping half the gifts failed to improve the solution
        indexes = np.random.permutation(range(len(trip)))
        for index in indexes:
          neighbor = OptimalMoveGiftInTripNeighbor(trips, self.log, trip=i, gift_index=index)
          # neighbor = OptimalSwapInRandomTripNeighbor(trips, self.log, trip=trip, first_gift=index)
          if neighbor.cost_delta < 0:
            improvement += neighbor.cost_delta
            current_improvement += neighbor.cost_delta
            swaps += 1
            current_swaps += 1
            neighbor.apply()
          else:
            no_improvement_count += 1
            current_no_improvement_count += 1
            if current_no_improvement_count - current_swaps > abort_after_x_without_improvement:
              break

      self.log.debug("Checked {:>3d} in {} tries for {:>3d}-gift trip: swapped {:>2d} gifts with an improvement of {:.3}M".format(
        swaps + no_improvement_count, tries, len(trip), swaps, improvement / 1e6))

      # done processing the trip, write back the permutated trip
      trips[i] = trip

    # extract gift/trip mapping
    combined_trips = np.concatenate(trips)[:, [utils.GIFT, utils.TRIP]]
    self.trips = pd.DataFrame(combined_trips, columns=["GiftId", "TripId"])

