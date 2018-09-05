"""
Partition an integer.
"""


from .memoize import LookUpTable


def yield_partition(n):
    """
    @returns Unique partitions of integer as iterable
    """
    a = [0] * (n + 1)
    k = 1
    y = n - 1

    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y += x - 1
        yield tuple(a[:k + 1])

@LookUpTable
def partition(n):
    """
    @returns Unique partitions of integer as tuple
    """
    return tuple(list(yield_partition(n)))

if __name__ == "__main__":
    print list(yield_partition(5))
