#!/usr/bin/env python

import copy
import utils
import glob
import math
import pandas as pd
import numpy as np
from method import Method
from haversine import haversine


class SimulatedAnnealingTripMethod(Method):
  @property
  def name(self):
    return "sim-trip"

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

  def calcTripCost(self, trip):
      #print(trip)
      weight = sum(trip[:, 4]) + utils.SLEIGH_WEIGHT
      #print("weight", weight)
      cost = 0
      
      lat = 2
      long = 3
      w = 4
      
      #print("lat", trip[0,lat])
      #print("long", trip[0,long])
      #print("w", trip[0,w])
      
      cost = haversine(utils.NORTH_POLE, (trip[0,lat], trip[0,long])) * weight
      
      for i, gift in enumerate(trip):
          weight -= gift[w]

          if i == trip.shape[0]-1:
              break
          
          cost += haversine((trip[i,lat], trip[i,long]), (trip[i+1,lat], trip[i+1,long]))*weight
          
      cost += haversine(utils.NORTH_POLE, (trip[-1,lat], trip[-1,long])) * weight
      
      #print("empty weight", weight)
      
      return cost

  def run(self, args):
    """
    idea: optimize the trip's routes within an existing solution
    using simmulated annealing
    """
    matches = glob.glob("data/*{}*".format(args.from_file))
    if not matches:
        self.log.warning("Not matching file found, aborting!")
        return
    
    all_trips = pd.read_csv(matches[0]).merge(self.gifts, on="GiftId")
    
    
    trips = [all_trips[all_trips.TripId == t].values for t in all_trips.TripId.unique()]
    #print("trips", trips)
    #print("last trip", trips[-1])
    
    
    startTemperature = 100
    alpha = 0.99
    roundsPerTemperature = 5;
    minTemperature = 5
    for i, trip in enumerate(trips):
        #print(type(trip))
        temperature = startTemperature
        currentSolution = copy.deepcopy(trip)
        bestTrip = copy.deepcopy(trip)
        #print(type(bestTrip))
        bestTripCost = self.calcTripCost(bestTrip)

#         if not set(bestTrip[:,0]) == set(trip[:,0]):
#             print("wtf")
#             print(set(bestTrip[:,0]))
#             print("------")
#             print(set(trip[:,0]))
        
        # self.log.debug(cost)
        while temperature > minTemperature:
            for round in range(roundsPerTemperature):
                workingTrip = copy.deepcopy(currentSolution)
                
                cost = self.calcTripCost(currentSolution)
                
                # select randomly two points
                a = np.random.randint(currentSolution.shape[0])
                b = a
                while (a == b):
                    b = np.random.randint(currentSolution.shape[0])
            
            
    #             if not set(workingTrip[:,0]) == set(trip[:,0]):
    #                 print("before swap")
    #                 print(set(workingTrip[:,0]))
    #                 print("------")
    #                 print(set(trip[:,0]))
                tmp = copy.deepcopy(workingTrip[a])
                workingTrip[a] = workingTrip[b]
                workingTrip[b] = tmp
    #             if not set(workingTrip[:,0]) == set(trip[:,0]):
    #                 print("after swap ({}({}) <-> {}({}))".format(a, currentSolution[a,0], b, currentSolution[b,0]))
    #                 print(set(workingTrip[:,0]))
    #                 print("------")
    #                 print(set(trip[:,0]))
    #                 return
                    
                #3print(workingTrip[a], currentSolution[a]);
                #workingTrip[a], workingTrip[b] = workingTrip[b], workingTrip[a]
                #print(workingTrip[a], currentSolution[a]);
    
                
                newCost = self.calcTripCost(workingTrip)
                
                useNew = False
            
                if newCost < cost :
                    useNew = True
                else :
                    rand = np.random.random()    
                    delta = newCost - cost
                    calc = math.exp(-delta / temperature)
                    if calc > rand:
                        useNew = True
                        
                if useNew:
#                     if not set(currentSolution[:,0]) == set(workingTrip[:,0]):
#                         print("error at {}, lost giftids".format(i))
#                         return
                    currentSolution = copy.deepcopy(workingTrip)
                    if newCost < bestTripCost:
                        #self.log.debug("Found new best at temperature {}".format(temperature))
                        bestTrip = copy.deepcopy(workingTrip)
                        bestTripCost = newCost
            
            
            temperature *= alpha
            
        if (i+1) % 100 == 0:
            self.log.info("optimized trip {} of {}".format(i+1, len(trips)))
        trips[i] = copy.deepcopy(bestTrip)
    
    combined_trips = np.concatenate(trips)[:, [0, 1]]
    self.trips = pd.DataFrame(combined_trips, columns=["GiftId", "TripId"])
    cols = ["GiftId","TripId"]
    self.trips[cols] = self.trips[cols].applymap(np.int64)
    #print("self.trips", self.trips)
        