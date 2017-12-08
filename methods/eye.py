#!/usr/bin/env python

# https://www.kaggle.com/the1owl/you-ll-shoot-your-eye-out-kid

import sqlite3

import pandas as pd

import utils
from haversine import haversine
from method import Method


class EyeMethod(Method):
  def bb_sort(self, ll):
    s_limit = 5000
    optimal = False
    ll = [[0,utils.NORTH_POLE,10]] + ll[:] + [[0,utils.NORTH_POLE,10]]
    while not optimal:
      optimal = True
      for i in range(1,len(ll) - 2):
        lcopy = ll[:]
        lcopy[i], lcopy[i+1] = ll[i+1][:], ll[i][:]
        if self.path_opt_test(ll[1:-1]) > self.path_opt_test(lcopy[1:-1]):
          #print("swap")
          ll = lcopy[:]
          optimal = False
          s_limit -= 1
          if s_limit < 0:
            optimal = True
            break
    return ll[1:-1]

  def path_opt_test(self, llo):
    f_ = 0.0
    d_ = 0.0
    l_ = utils.NORTH_POLE
    for i in range(len(llo)):
      d_ += haversine(l_, llo[i][1])
      f_ += d_ * llo[i][2]
      l_ = llo[i][1]
    d_ += haversine(l_, utils.NORTH_POLE)
    f_ += d_ * 10 #sleigh weight for whole trip
    return f_

  @property
  def name(self):
    return "eye"

  def run(self, args):
    c = sqlite3.connect(":memory:")
    self.gifts.to_sql("gifts",c)
    cu = c.cursor()
    cu.execute("ALTER TABLE gifts ADD COLUMN 'TripId' INT;")
    cu.execute("ALTER TABLE gifts ADD COLUMN 'i' INT;")
    cu.execute("ALTER TABLE gifts ADD COLUMN 'j' INT;")
    c.commit()

    for n in [1.25252525]:
      i_ = 0
      j_ = 0
      for i in range(90,-90,int(-180/n)):
        i_ += 1
        j_ = 0
        for j in range(180,-180,int(-360/n)):
          j_ += 1
          cu = c.cursor()
          cu.execute("UPDATE gifts SET i=" + str(i_) + ", j=" + str(j_) + " WHERE ((Latitude BETWEEN " + str(i - (180/n)) + " AND  " + str(i) + ") AND (Longitude BETWEEN " + str(j - (360/n)) + " AND  " + str(j) + "));")
          c.commit()

      for limit_ in [67]:
        trips = pd.read_sql("SELECT * FROM (SELECT * FROM gifts WHERE TripId IS NULL ORDER BY i, j, Longitude, Latitude LIMIT " + str(limit_) + " ) ORDER BY Latitude DESC",c)
        t_ = 0
        while len(trips.GiftId)>0:
          g = []
          t_ += 1
          w_ = 0.0
          for i in range(len(trips.GiftId)):
            if (w_ + float(trips.Weight[i]))<= utils.WEIGHT_LIMIT:
              w_ += float(trips.Weight[i])
              g.append(trips.GiftId[i])
          cu = c.cursor()
          cu.execute("UPDATE gifts SET TripId = " + str(t_) + " WHERE GiftId IN(" + (",").join(map(str,g)) + ");")
          c.commit()

          trips = pd.read_sql("SELECT * FROM (SELECT * FROM gifts WHERE TripId IS NULL ORDER BY i, j, Longitude, Latitude LIMIT " + str(limit_) + " ) ORDER BY Latitude DESC",c)
          #break

        # ou_ = open("submission_opt" + str(limit_) + " " + str(n) + ".csv","w")
        # ou_.write("TripId,GiftId\n")
        bm = 0.0
        submission = pd.read_sql("SELECT TripId FROM gifts GROUP BY TripId ORDER BY TripId;", c)
        trips = []
        for s_ in range(len(submission.TripId)):
          trip = pd.read_sql("SELECT GiftId, Latitude, Longitude, Weight FROM gifts WHERE TripId = " + str(submission.TripId[s_]) + " ORDER BY Latitude DESC, Longitude ASC;",c)
          a = []
          for x_ in range(len(trip.GiftId)):
            a.append([trip.GiftId[x_],(trip.Latitude[x_],trip.Longitude[x_]),trip.Weight[x_]])
          b = self.bb_sort(a)
          if self.path_opt_test(a) <= self.path_opt_test(b):
            print(submission.TripId[s_], "No Change", self.path_opt_test(a) , self.path_opt_test(b))
            bm += self.path_opt_test(a)
            for y_ in range(len(a)):
              trips.append({"TripId": submission.TripId[s_], "GiftId": a[y_][0]})
              # ou_.write(str(submission.TripId[s_])+","+str(a[y_][0])+"\n")
          else:
            print(submission.TripId[s_], "Optimized", self.path_opt_test(a) - self.path_opt_test(b))
            bm += self.path_opt_test(b)
            for y_ in range(len(b)):
              trips.append({"TripId": submission.TripId[s_], "GiftId": b[y_][0]})
              # ou_.write(str(submission.TripId[s_])+","+str(b[y_][0])+"\n")
        # ou_.close()
        self.trips = self.trips.append(trips)

        benchmark = 12514008574.2
        if bm < benchmark:
          print(n, limit_, "Improvement", bm, bm - benchmark, benchmark)
        else:
          print(n, limit_, "Try again", bm, bm - benchmark, benchmark)
        cu = c.cursor()
        cu.execute("UPDATE gifts SET TripId = NULL;")
        c.commit()
