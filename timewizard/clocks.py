"""Rendered training clocks from SynClock, the `synclock` git dependency built
from our fork of itsabouttime."""

from __future__ import annotations

import random

import cv2
import numpy as np
import SynClock
from PIL import Image

from timewizard.reading import Time

# Fork-based dataset workers die when OpenCV keeps a thread pool.
cv2.setNumThreads(0)


def sample(rng: random.Random) -> tuple[Image.Image, Time]:
    """SynClock draws from module-level random state, so it is reseeded per call."""
    random.seed(rng.getrandbits(32))
    np.random.seed(rng.getrandbits(32))
    image, details, _ = SynClock.gen_clock(return_details=True)
    hour = int(details["hour"])
    return Image.fromarray(image[:, :, ::-1]), Time(hours=hour or 12, minutes=int(details["minute"]))
