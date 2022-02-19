""" Prediction models
"""

from models.networks import get_network_from_options
from .map_predictor_model import OccupancyPredictor


def get_predictor_from_options(options):
    return OccupancyPredictor(segmentation_model=get_network_from_options(options),
                            map_loss_scale=options.map_loss_scale)