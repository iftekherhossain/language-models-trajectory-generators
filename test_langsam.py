from env import Environment, run_simulation_environment
import argparse
from multiprocessing import Pipe, log_to_stderr,Process
import logging
import time
from api import API
import openai
import os
import torch
from lang_sam import LangSAM
import sys
import functools
import config
import random
import gc
sys.path.append("./XMem/")
print = functools.partial(print, flush=True)

from XMem.model.network import XMem
sc_factor = 0.08
object_list = [
    ("002_master_chef_can", sc_factor, "master chef can"),
    ("003_cracker_box", sc_factor, "cracker box"),
    ("004_sugar_box", sc_factor, "sugar box"),
    ("005_tomato_soup_can", sc_factor + 0.1, "tomato soup can"),
    ("006_mustard_bottle", sc_factor + 0.05, "mustard bottle"),
    ("007_tuna_fish_can", sc_factor + 0.1, "tuna fish can"),
    ("008_pudding_box", sc_factor, "pudding box"),
    ("009_gelatin_box", sc_factor, "gelatin box"),
    ("010_potted_meat_can", sc_factor + 0.1, "potted meat can"),
    ("011_banana", sc_factor + 0.05, "banana"),
    ("012_strawberry", sc_factor, "strawberry"),
    ("013_apple", sc_factor, "apple"),
    ("014_lemon", sc_factor, "lemon"),
    ("015_peach", sc_factor, "peach"),
    ("016_pear", sc_factor, "pear"),
    ("017_orange", sc_factor, "orange"),
    ("018_plum", sc_factor, "plum"),
    ("019_pitcher_base", sc_factor, "jug"),
    ("021_bleach_cleanser", sc_factor, "bleach cleanser"),
    ("022_windex_bottle", sc_factor, "windex bottle"),
    ("024_bowl", sc_factor, "bowl"),
    ("025_mug", sc_factor, "mug"),
    ("026_sponge", sc_factor, "sponge"),
    ("028_skillet_lid", sc_factor, "skillet lid"),
    ("029_plate", sc_factor, "plate"),
    ("030_fork", sc_factor+0.1, "fork"),
    ("031_spoon", sc_factor+0.1, "spoon"),
    ("032_knife", sc_factor+0.1, "knife"),
    ("033_spatula", sc_factor, "spatula"),
    ("035_power_drill", sc_factor, "power drill"),
    ("036_wood_block", sc_factor, "wood block"),
    ("037_scissors", sc_factor, "scissors"),
    ("038_padlock", sc_factor+0.2, "padlock"),
    ("040_large_marker", sc_factor + 0.2, "large marker"),
    ("042_adjustable_wrench", sc_factor, "adjustable wrench"),
    ("043_phillips_screwdriver", sc_factor+0.05, "phillips screwdriver"),
    ("044_flat_screwdriver", sc_factor + 0.12, "flat screwdriver"),
    ("048_hammer", sc_factor, "hammer"),
    ("050_medium_clamp", sc_factor+0.15, "medium clamp"),
    ("051_large_clamp", sc_factor + 0.05, "large clamp"),
    ("052_extra_large_clamp", sc_factor, "extra large clamp"),
    ("053_mini_soccer_ball", sc_factor, "mini soccer ball"),
    ("054_softball", sc_factor, "softball"),
    ("055_baseball", sc_factor, "baseball"),
    ("056_tennis_ball", sc_factor, "tennis ball"),
    ("057_racquetball", sc_factor+0.2, "racquetball"),
    ("058_golf_ball", sc_factor, "golf ball"),
    ("059_chain", sc_factor, "chain"),
    ("061_foam_brick", sc_factor+0.25, "foam brick"),
    ("062_dice", sc_factor+0.25, "dice"),
    ("063-a_marbles", sc_factor, "marbles"),  # use only one from 063
    ("065-a_cups", sc_factor+0.2, "cups"),        # use only one from 065
    ("070-a_colored_wood_blocks", sc_factor, "colored wood blocks"),
    ("071_nine_hole_peg_test", sc_factor, "nine hole peg test"),
    ("072-a_toy_airplane", sc_factor, "toy airplane"),
    ("073-a_lego_duplo", sc_factor+0.25, "lego duplo"),
    ("077_rubiks_cube", sc_factor+0.1, "rubiks cube")
]

while True:

    selected_objects = random.sample(object_list, 3)
    print(selected_objects)
    parser = argparse.ArgumentParser(description="Main Program.")
    parser.add_argument("-lm", "--language_model", choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], default="gpt-4o-mini", help="select language model")
    parser.add_argument("-r", "--robot", choices=["sawyer", "franka"], default="franka", help="select robot")
    parser.add_argument("-m", "--mode", choices=["default", "debug"], default="debug", help="select mode to run")
    parser.add_argument(
        "-o",
        "--objects", 
        nargs="+",  # Accept one or more arguments
        type=str, 
        default=selected_objects,
        help="List of YCB object names with scaling factor, e.g. --objects 025_mug:1.0 032_knife:0.8"
    )
    args = parser.parse_args()

    logger = log_to_stderr()
    logger.setLevel(logging.INFO)
    main_conn, environment_conn = Pipe()


    # Device
    if torch.cuda.is_available():
        logger.info("Using GPU.")
        device = torch.device("cuda")
    else:
        logger.info("CUDA not available. Please connect to a GPU instance if possible.")
        device = torch.device("cpu")

    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI()
    print("Kugi1")
    
    env_process = Process(target=run_simulation_environment, name="EnvProcess", args=[args, environment_conn, logger])
    env_process.start()
    time.sleep(3)  # Optional delay before next loop iteration
    print("Kugi2")
    langsam_model = LangSAM()
    xmem_model = XMem(config.xmem_config, "./XMem/saves/XMem.pth", device).eval().to(device)
    api = API(args, main_conn, logger, client, langsam_model, xmem_model, device)

    [env_connection_message] = main_conn.recv()
    logger.info(env_connection_message)
    detect_object = api.detect_object
    

    name = random.sample(selected_objects,1)[0][-1]
    print("Kugi3------------------------------------------------------------------------------------",name)
    detect_object(name)
    
    # time.sleep(5)
    print("✅ Detection finished, closing environment...")

    # Cleanly shut down environment and communication pipes
    env_process.terminate()
    env_process.join()
    main_conn.close()
    environment_conn.close()

    print("🛑 Environment closed. Looping for next batch...\n")
    time.sleep(2)  # Optional delay before next loop iteration
    torch.cuda.empty_cache()
    gc.collect()