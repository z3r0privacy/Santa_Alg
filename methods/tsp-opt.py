#!/usr/bin/env python

import numpy as np

import pandas as pd
import utils
from method import Method
from neighbors import SwapGiftsInTripNeighbor


class TspOptimizeTripMethod(Method):
  def __init__(self, gifts, log):
    super(TspOptimizeTripMethod, self).__init__(gifts, log)

  @property
  def name(self):
    return "triptsp"

  def run(self, args):
    """
    Idea: Load existing valid solution and improve it by improving the order of gift distribution
    within a single trip. Change order by randomly selecting x locations and swapping them with the
    one that provides the largest decrease in cost.
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
      for foobar in range(swaps_per_trip):
        index = np.random.randint(trip.shape[0])
        neighbor = None

        for j in range(trip.shape[0]):
          if j == index:
            continue
          # this is broken (signature change in constructor)
          new_neighbor = SwapGiftsInTripNeighbor(trip, index, j, self.log)
          if not neighbor or new_neighbor.cost_delta < neighbor.cost_delta:
            neighbor = new_neighbor

        if neighbor.cost_delta >= 0:
          # self.log.debug("No improvement found from swapping {}...".format(index))
          continue
        total_improvement += neighbor.cost_delta
        swaps += 1
        neighbor.apply()

      self.log.debug("Swapped {} locations with a total improvement of {}".format(swaps, total_improvement))

      # done processing the trip, write back the permutated trip
      trips[i] = trip

    # extract gift/trip mapping
    combined_trips = np.concatenate(trips)[:, [utils.GIFT, utils.TRIP]]
    self.trips = pd.DataFrame(combined_trips, columns=["GiftId", "TripId"])

