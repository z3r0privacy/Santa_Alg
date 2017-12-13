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
    log.info("Optimizing trips...")
    method.run(args)
    if method.verify_trips():
      method.evaluate_trips()
      method.write_trips(solution_file)
    else:
      method.write_trips("last-failure.csv")
  except Exception as ex:
    method.write_trips("last-failure.csv")
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
  log = utils.get_logger("santa-sleigh")
  gifts = pd.read_csv("data/gifts.csv")

  methods = {method.name: method for method in get_all_methods(gifts, log)}

  parser = argparse.ArgumentParser()

  # general arguments
  parser.add_argument("method", choices=methods.keys(), help="method to try")

  # method-specific arguments
  parser.add_argument("--from-file", required=False, help=
      "Pattern to match files under 'data/' against which contains a solution to load as basis for the new evaluation")

  args = parser.parse_args()
  evaluation_id = "{}-{}".format(args.method, datetime.utcnow().strftime("%Y-%m-%d-%H:%M:%S"))
  args.evaluation_id = evaluation_id
  solution_file = "solutions/{}.csv".format(evaluation_id)
  log.debug("method: '{}', args: '{}'".format(args.method, args))

  main(methods[args.method], solution_file, args)
