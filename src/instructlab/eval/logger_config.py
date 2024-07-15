# SPDX-License-Identifier: Apache-2.0
# Standard
import logging


def setup_logger(name):
    # Set up the logger
    logger = logging.getLogger(name)
    return logger
