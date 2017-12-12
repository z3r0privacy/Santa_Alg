#!/usr/bin/env python

import numpy as np
import pandas as pd

import utils
from method import Method
from neighbor import Neighbor
from neighbors import (MoveGiftToAnotherTripNeighbor,
                       MoveGiftToLightestTripNeighbor,
                       SplitOneTripIntoTwoNeighbor,
                       SwapGiftsAcrossTripsNeighbor, SwapGiftsInTripNeighbor)


class SimulatedAnnealingMethod(Method):
  @property
  def name(self):
    return "sim"

  def _get_neighbors(self, trips):
    number_of_same_neighbor = 1

    # return [MoveGiftToAnotherTripNeighbor(trips, self.log) for i in range(number_of_same_neighbor)]
    # return [MoveGiftToLightestTripNeighbor(trips, self.log) for i in range(number_of_same_neighbor)]

    return np.random.permutation(np.array(
      [[neighbor(trips, self.log) for i in range(number_of_same_neighbor)]
        for neighbor in Neighbor.__subclasses__()]
      ).flatten())

  def run(self, args):
    """
    Idea: Apply simulated annealing (d'uh).
    """
    # TODO: Get parameters from arguments
    all_trips = self._load_trips_from_file(args)
    if all_trips is None:
      return
    iterations = int(1e4)
    initial_temperature = 1e6
    alpha = 0.9
    moves = {}

    # TODO: Checkpoints

    temperature = initial_temperature

    # split all trips into separate trips
    trips = [all_trips[all_trips.TripId == t].values for t in all_trips.TripId.unique()]

    good_solutions = 0
    accepted_bad_solutions = 0
    rejected_bad_solutions = 0
    last_good_solutions = 0
    last_accepted_bad_solutions = 0
    last_rejected_bad_solutions = 0
    last_cost_change = 0
    total_cost_change = 0
    log_interval = int(1e3)
    for i in range(iterations):
      if i > 0 and i % log_interval == 0:
        self.log.debug("{:>6}/{}: T={:>9.1f}, since {:>6}: {:>4.1f}/{:>4.1f}/{:>4.1f}% good/acc/rej, cost: {:>9.1f}k".format(
          i, iterations, temperature, i - log_interval,
          100.0 * (good_solutions - last_good_solutions) / log_interval,
          100.0 * (accepted_bad_solutions - last_accepted_bad_solutions) / log_interval,
          100.0 * (rejected_bad_solutions - last_rejected_bad_solutions) / log_interval,
          (total_cost_change - last_cost_change) / 1e3))
        last_cost_change = total_cost_change
        last_good_solutions = good_solutions
        last_accepted_bad_solutions = accepted_bad_solutions
        last_rejected_bad_solutions = rejected_bad_solutions

      # decrease temperature after every x solutions
      if i > 0 and i % 1e2 == 0:
        temperature *= alpha

      # select neighbor - try all neighbors to find any good (or the least bad) neighbor
      neighbors = self._get_neighbors(trips)
      neighbor = best_bad_neighbor = neighbors[0]

      # print(neighbor.cost_delta, neighbor)

      # for j in range(1+0*len(neighbors)):
      #   neighbor = neighbors[j]
      #   if neighbor.cost_delta < 0:
      #     break
      #   if best_bad_neighbor.cost_delta > neighbor.cost_delta:
      #     best_bad_neighbor = neighbor

      if neighbor.cost_delta < 0:
        # self.log.success("Accepting neighbor {} with negative cost {}".format(neighbor, neighbor.cost_delta))
        total_cost_change += neighbor.cost_delta
        neighbor.apply()
        good_solutions += 1
        if not neighbor.__class__.__name__ in moves.keys():
          moves[neighbor.__class__.__name__] = 0
        moves[neighbor.__class__.__name__] += 1
        self.check_gifts(trips, neighbor)
        continue

      accepting_probability = np.exp(-best_bad_neighbor.cost_delta/temperature)
      if accepting_probability > np.random.rand():
        # self.log.info("Accepting worse neighbor {:>20} (by {:>9.1f}, {:>4.1f}% chance, T={:>9.1f})".format(
        #   str(best_bad_neighbor), best_bad_neighbor.cost_delta, 100 * accepting_probability, temperature))
        total_cost_change += best_bad_neighbor.cost_delta
        best_bad_neighbor.apply()
        accepted_bad_solutions += 1
        if not neighbor.__class__.__name__ in moves.keys():
          moves[neighbor.__class__.__name__] = 0
        moves[neighbor.__class__.__name__] += 1
        self.check_gifts(trips, neighbor)
      else:
        # self.log.debug("Rejecting worse neighbor {:>20} (by {:>9.1f}, {:>4.1f}% chance, T={:>9.1f})".format(
        #   str(best_bad_neighbor), best_bad_neighbor.cost_delta, 100 * accepting_probability, temperature))
        rejected_bad_solutions += 1

      # increase temperature after every x bad solutions
      if (accepted_bad_solutions + rejected_bad_solutions) % 5000 == 0:
        temperature = (3*temperature + initial_temperature)/4

    self.log.info("Finished {} iterations with total cost change {}".format(iterations, total_cost_change))
    self.log.info("Evaluated neighbors: {} good, {} accepted/{} rejected bad solutions ({:.1f}/{:.1f}/{:.1f}%)".format(
      good_solutions, accepted_bad_solutions, rejected_bad_solutions,
      100.0 * good_solutions / iterations, 100.0 * accepted_bad_solutions / iterations, 100 * rejected_bad_solutions / iterations))
    self.log.info("Applied the following moves: {}".format(moves))
    # extract gift/trip mapping
    combined_trips = np.concatenate(trips)[:, [0, 1]]
    self.trips = pd.DataFrame(combined_trips, columns=["GiftId", "TripId"])

  def check_gifts(self, trips, neighbor, print_weights=False):
    if not Neighbor.VERIFY_COST_DELTA:
      return

    for trip in trips:
      if len(set(trip[:, utils.TRIP])) != 1:
        self.log.error("Trip has gifts with unexpected number of trip IDs: {} (previous move: {})".format(set(trip[:, utils.TRIP]), str(neighbor)))
        print(trip[:, utils.TRIP])
        raise ValueError()

    weights = [np.sum(t[:, utils.WEIGHT]) for t in trips]
    if print_weights:
      print("weights", np.max(weights))
    if np.max(weights) > utils.WEIGHT_LIMIT:
      self.log.error("Trip with invalid weight: {} (previous move: {})".format(np.max(weights), str(neighbor)))
      raise ValueError()

    gifts = [t[:, utils.GIFT] for t in trips]
    flattened = [val for sublist in gifts for val in sublist]
    gifts = np.unique(flattened)
    if len(gifts) != 100000:
      self.log.error("Wrong number of gifts: {} (previous move: {})".format(len(gifts), str(neighbor)))
      raise ValueError()


