#!/usr/bin/env python

import gc
from multiprocessing import Pool

import numpy as np
import pandas as pd

import utils
from method import Method
from neighbor import Neighbor
from neighbors import (MoveGiftToAnotherTripNeighbor,
                       MoveGiftToLightestTripNeighbor,
                       MoveGiftToOptimalTripNeighbor,
                       OptimalHorizontalTripSplitNeighbor,
                       OptimalMergeTripIntoAdjacentNeighbor,
                       OptimalMoveGiftInTripNeighbor,
                       OptimalSwapInRandomTripNeighbor,
                       OptimalVerticalTripSplitNeighbor,
                       SplitOneTripIntoTwoNeighbor,
                       SwapGiftsAcrossTripsNeighbor,
                       SwapRandomGiftsInTripNeighbor)


class SimulatedAnnealingMethod(Method):
  @property
  def name(self):
    return "sim"

  def _get_neighbors(self, trips):
    # current neighbor to test
    # return [OptimalMergeTripIntoAdjacentNeighbor(trips, self.log)]

    # optimum neighbors in descending order of calculation complexity
    # TODO: Try different weights per neighbor and different weights for trips within neighbors (based on cost/weight)
    return [
        # TODO: Specify more restrictive heuristic restrictions in neighbors

        # MERGE-TRIP NEIGHBORHOOD
        # 1000 iterations:
        # merge current trip into neighbors
        OptimalMergeTripIntoAdjacentNeighbor(trips),

        # TWO-TRIP NEIGHBORHOOD
        # 1000 iterations: 6m11s
        MoveGiftToOptimalTripNeighbor(trips),

        # NEW-TRIP NEIGHBORHOOD
        # 1000 iterations: 1m15s
        OptimalHorizontalTripSplitNeighbor(trips),
        # 1000 iterations: 1m20s
        OptimalVerticalTripSplitNeighbor(trips),

        # SINGLE-TRIP NEIGHBORHOOD
        # 1000 iterations: 1m44s
        OptimalSwapInRandomTripNeighbor(trips),
        # 1000 iterations single-threaded: 1m51s
        # 1000 iterations on four threads: 2m6s
        # 1000 iterations on two threads:  2m10s
        # 1000 iterations on two threads with length > 50:  1m49s
        # 1000 iterations on two threads with length > 100: 1m43s
        # 1000 iterations on two threads with length > 200: 1m45s
        # 1000 iterations on two threads with length > 500: 1m44s
        OptimalMoveGiftInTripNeighbor(trips),
        ]

  def run(self, args):
    """
    Idea: Apply simulated annealing (d'uh).
    """
    all_trips = self._load_trips_from_file(args)
    if all_trips is None:
      return

    iterations = int(1e2)
    log_interval = int(1e1)
    checkpoint_interval = int(1e3)
    worker_size = 2
    log_neighbors = False

    # hyperparameters
    initial_temperature = args.temperature or 1e5
    temperature_decrease = iterations / 1e1
    alpha = args.alpha or 0.9

    moves = {}
    temperature = initial_temperature

    # split all trips into separate trips
    trips = [all_trips[all_trips.TripId == t].values for t in all_trips.TripId.unique()]

    # variables for stats
    good_solutions = 0
    accepted_bad_solutions = 0
    rejected_bad_solutions = 0
    last_good_solutions = 0
    last_accepted_bad_solutions = 0
    last_rejected_bad_solutions = 0
    last_cost_change = 0
    total_cost_change = 0

    self.log.info("Parameters: {} iterations, logs every {}, checkpoints every {}; {} workers; T={}, decrease every {}, alpha={}".format(
      iterations, log_interval, checkpoint_interval, worker_size, initial_temperature, temperature_decrease, alpha))

    with Pool(worker_size) as pool:
      for i in range(iterations+1):
        if i > 0 and i % log_interval == 0:
          self.log.debug("{:>6}/{}: T={:>9.1f}, since {:>6}: {:>4.1f}/{:>4.1f}/{:>4.1f}% good/acc/rej ({:>2.1f}% acc), cost: {:>9.1f}k/{:.1f}M".format(
            i, iterations, temperature, i - log_interval,
            100.0 * (good_solutions - last_good_solutions) / log_interval,
            100.0 * (accepted_bad_solutions - last_accepted_bad_solutions) / log_interval,
            100.0 * (rejected_bad_solutions - last_rejected_bad_solutions) / log_interval,
            100.0 * (accepted_bad_solutions - last_accepted_bad_solutions) / (accepted_bad_solutions - last_accepted_bad_solutions + rejected_bad_solutions - last_rejected_bad_solutions),
            (total_cost_change - last_cost_change) / 1e3, total_cost_change / 1e6))
          if i == iterations:
            break
          last_cost_change = total_cost_change
          last_good_solutions = good_solutions
          last_accepted_bad_solutions = accepted_bad_solutions
          last_rejected_bad_solutions = rejected_bad_solutions
          print("gc")
          gc.collect()

        if i > 0 and i % checkpoint_interval == 0:
          if not self.create_checkpoint(trips, i, iterations, args.evaluation_id, args.random_seed):
            self.log.error("Aborting evaluation because the current solution is invalid")
            break

        # decrease temperature after every x solutions
        if i > 0 and i % temperature_decrease == 0:
          temperature *= alpha

        # select neighbor
        neighbors = self._get_neighbors(trips)
        jobs = []
        for neighbor in neighbors:
          jobs.append(pool.apply_async(neighbor.cost_delta))
        costs = [j.get() for j in jobs]
        neighbor = neighbors[costs.index(min(costs))]
        neighbor_name = neighbor.__class__.__name__

        if neighbor.cost_delta() < 0:
          if log_neighbors:
            self.log.success("Accepting neighbor {} with negative cost {:.1f}k".format(neighbor, neighbor.cost_delta() / 1e3))
          total_cost_change += neighbor.cost_delta()
          neighbor.apply()
          good_solutions += 1
          if not neighbor_name in moves.keys():
            moves[neighbor_name] = {}
          if not "good" in moves[neighbor_name].keys():
            moves[neighbor_name]["good"] = 0
          moves[neighbor_name]["good"] += 1
          self.check_gifts(trips, neighbor)
          continue

        accepting_probability = np.exp(-neighbor.cost_delta()/temperature)
        if accepting_probability > np.random.rand():
          if log_neighbors:
            self.log.info("Accepting worse neighbor {:>20} (by {:>.1f}k, {:>4.1f}% chance, T={:>9.1f})".format(
              str(neighbor), neighbor.cost_delta() / 1e3, 100 * accepting_probability, temperature))
          total_cost_change += neighbor.cost_delta()
          neighbor.apply()
          accepted_bad_solutions += 1
          if not neighbor_name in moves.keys():
            moves[neighbor_name] = {}
          if not "acc" in moves[neighbor_name].keys():
            moves[neighbor_name]["acc"] = 0
          moves[neighbor_name]["acc"] += 1
          self.check_gifts(trips, neighbor)
        else:
          if log_neighbors:
            self.log.debug("Rejecting worse neighbor {:>20} (by {:>.1f}k, {:>4.1f}% chance, T={:>9.1f})".format(
              str(neighbor), neighbor.cost_delta() / 1e3, 100 * accepting_probability, temperature))
          rejected_bad_solutions += 1
          if not neighbor_name in moves.keys():
            moves[neighbor_name] = {}
          if not "rej" in moves[neighbor_name].keys():
            moves[neighbor_name]["rej"] = 0
          moves[neighbor_name]["rej"] += 1

    self.log.info("Finished {} iterations with total cost change of {}M".format(iterations, total_cost_change / 1e6))
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

    for i, trip in enumerate(trips):
      if len(set(trip[:, utils.TRIP])) != 1:
        self.log.error("Trip has gifts with unexpected number of trip IDs: {} (previous move: {})".format(set(trip[:, utils.TRIP]), str(neighbor)))
        print(i, trip[:, utils.TRIP])
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

  def create_checkpoint(self, trips, i, iterations, evaluation_id, random_seed):
    checkpoint_file = "checkpoints/{}_{}_{}.csv".format(evaluation_id, random_seed, i)
    self.log.info("{:>6}/{}: Creating checkpoint '{}'".format(i, iterations, checkpoint_file))

    # extract gift/trip mapping
    combined_trips = np.concatenate(trips)[:, [0, 1]]
    self.trips = pd.DataFrame(combined_trips, columns=["GiftId", "TripId"])
    self.write_trips(checkpoint_file)

    return self.verify_trips()

