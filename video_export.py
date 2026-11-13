"""Shared export settings for the MP4 encoders."""

from fractions import Fraction

DEFAULT_MAX_HEIGHT = 480
DEFAULT_CRF = 16


def scale_filter(max_height):
    # yuv420p needs even dimensions. Keep an already-even native size below the cap.
    target = "trunc(ih/2)*2" if not max_height else "min({},trunc(ih/2)*2)".format(max_height)
    width = "trunc(iw/2)*2" if not max_height else "if(gt(ih,{}),-2,trunc(iw/2)*2)".format(max_height)
    return "scale=w='{}':h='{}':flags=lanczos".format(width, target)


def export_size(width, height, max_height):
    if max_height and height > max_height:
        return max(2, 2 * round(Fraction(width * max_height, 2 * height))), max_height
    return max(2, width - width % 2), max(2, height - height % 2)
