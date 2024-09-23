import os

current = os.path.dirname(os.path.realpath(__file__))
BASE_PROYECT_PATH = os.path.abspath(os.path.join(current, "..", ".."))
DATASET_PATH = os.path.abspath(os.path.join(BASE_PROYECT_PATH, "dataset"))
PARTITIONS_PATH = os.path.abspath(os.path.join(BASE_PROYECT_PATH, "partitions"))
