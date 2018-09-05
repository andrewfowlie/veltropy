"""
Nearby-neighbour lookup.
"""

from functools import partial


class LookUpTable(dict):
    """
    Memoization by using a look-up table.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, other, *x):
        try:
            hash(x)
        except TypeError:
            return self.func(other, *x)

        try:
            table = self[other]
        except KeyError:
            table = self[other] = {}

        try:
            return table[x]
        except KeyError:
            result = table[x] = self.func(other, *x)
            return result

    def erase_lookup_table(self):
        """
        Erases look-up table.
        """
        self.clear()

def erase_lookup_tables(obj):
    """
    Erase any look-up tables on an object.
    """
    for class_ in obj.__class__.mro():
        for method in class_.__dict__.itervalues():
            try:
                method.erase_lookup_table()
            except AttributeError:
                pass
