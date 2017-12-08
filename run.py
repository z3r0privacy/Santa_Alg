#!/usr/bin/env python

import argparse
import logging
import sys
import time
import traceback

from datetime import datetime
import numpy as np
import pandas as pd

from method import Method
import methods
import utils


def main(method, solution_file, args):
  try:
    method.run(args)
    if method.evaluate_trips():
      method.write_trips(solution_file)
  except Exception as ex:
    log.critical("{} during evaluation: {}".format(type(ex).__name__, str(ex)))
    log.critical(traceback.format_exc())
    sys.exit(1)
  finally:
    mins, secs = divmod(time.time() - start_time, 60)
    hours, mins = divmod(mins, 60)
    time_string = \
        "{}h{}m{:.1f}s".format(int(hours), int(mins), secs) if hours > 0 else \
        "{}m{:.1f}s".format(int(mins), secs) if mins > 0 else \
        "{:.1f}s".format(secs)
    log.warning("Finished execution after {}".format(time_string))
    logging.shutdown()

def get_all_methods(gifts, log):
  return [sub(gifts, log) for sub in Method.__subclasses__()]

if __name__ == "__main__":
  start_time = time.time()

  np.random.seed(42)
  log = utils.get_logger("numerai")
  gifts = pd.read_csv("data/gifts.csv")

  methods = {method.name: method for method in get_all_methods(gifts, log)}

  parser = argparse.ArgumentParser()

  # general arguments
  parser.add_argument("method", choices=methods.keys(), help="method to try")

  # method-specific arguments
  # parser.add_argument("--exhaustive", required=False, action="store_true",
  #     help="True if the search should be exhaustive")

  args = parser.parse_args()
  solution_file = "solutions/{}-{}.csv".format(args.method, datetime.utcnow().strftime("%Y-%m-%d-%H:%M:%S"))
  log.debug("method: '{}', args: '{}'".format(args.method, args))

  main(methods[args.method], solution_file, args)
