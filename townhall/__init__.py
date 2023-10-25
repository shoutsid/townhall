"""
This file is used to initialize the app package.
"""

import logging
from .agents import *

# Set the root logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
