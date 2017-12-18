#!/usr/bin/env python

import numpy as np
import pandas as pd

import utils
from method import Method
from neighbors import (OptimalHorizontalTripSplitNeighbor,
    OptimalMoveGiftInTripNeighbor,
                       OptimalVerticalTripSplitNeighbor)


class CutExtremeTripsMethod(Method):
  def __init__(self, gifts, log):
    super(CutExtremeTripsMethod, self).__init__(gifts, log)

  @property
  def name(self):
    return "cutextremes"

  def run(self, args):
    """
    Idea: TBD
    """
    all_trips = self._load_trips_from_file(args)
    if all_trips is None:
      return

    # split all trips into separate trips
    trips = [all_trips[all_trips.TripId == t].values for t in all_trips.TripId.unique()]

    maximum_weight = 900
    maximum_length = 90
    large_trips = []
    for i, trip in enumerate(trips):
      if trip[:, utils.WEIGHT].sum() > maximum_weight and len(trip) > maximum_length:
        large_trips.append(i)

    self.log.info("Found {} trips with weight >{} and length >{}".format(len(large_trips), maximum_weight, maximum_length))

    total_cost_change = 0
    for large_trip_index in large_trips:
      horizontal_split = OptimalHorizontalTripSplitNeighbor(trips, large_trip_index)
      vertical_split = OptimalVerticalTripSplitNeighbor(trips, large_trip_index)
      neighbor = horizontal_split if horizontal_split.cost_delta() < vertical_split.cost_delta() else vertical_split
      previous_length = len(trips[large_trip_index])
      previous_weight = trips[large_trip_index][:, utils.WEIGHT].sum()
      neighbor.apply()
      self.log.debug("Applying {}\t(new trip gifts: {:.1f}%, cost: {:.3f}M) to trip {} (length {}, weight {:.1f})".format(
        neighbor, 100 * neighbor.first_trip_percentage,
        neighbor.cost_delta() / 1e6, large_trip_index, previous_length, previous_weight))
      total_cost_change += neighbor.cost_delta()

    self.log.info("Total cost of all modifications: {:.5f}M".format(total_cost_change / 1e6))

    # extract gift/trip mapping
    combined_trips = np.concatenate(trips)[:, [utils.GIFT, utils.TRIP]]
    self.trips = pd.DataFrame(combined_trips, columns=["GiftId", "TripId"])

