from env import Environment, run_simulation_environment
import argparse
from multiprocessing import Pipe, log_to_stderr,Process
import logging
import time
parser = argparse.ArgumentParser(description="Main Program.")
parser.add_argument("-lm", "--language_model", choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], default="gpt-4o-mini", help="select language model")
parser.add_argument("-r", "--robot", choices=["sawyer", "franka"], default="franka", help="select robot")
parser.add_argument("-m", "--mode", choices=["default", "debug"], default="debug", help="select mode to run")
args = parser.parse_args()

logger = log_to_stderr()
logger.setLevel(logging.INFO)
main, environment_conn = Pipe()

 # Start process
env_process = Process(target=run_simulation_environment, name="EnvProcess", args=[args, environment_conn, logger])
env_process.start()
time.sleep(5)