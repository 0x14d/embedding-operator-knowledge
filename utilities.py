import itertools
import re

from os import PathLike
from typing import Union

Path = Union[str, bytes, PathLike]


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def to_camel_case(snake_str):
    components = snake_str.split('_')
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + ''.join(x.title() for x in components[1:])


def camel_to_snake_ratings(name):
    """
     the influences in the database are saved as e.g. underExtrusion but the ratings are saved as under_extrusion
    that is why we have to change between camelCase and snake name convention until we have fixed it in the database
    TODO Jira issue XAI-568"""
    if name == 'overall':
        name = 'overall_ok'
    return camel_to_snake(name)
