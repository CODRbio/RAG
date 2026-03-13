import timeit
import logging

setup = """
from src.log.context import get_correlation_id as gci
"""

test1 = """
from src.log.context import get_correlation_id
get_correlation_id()
"""

test2 = """
gci()
"""

print("With import inside:", timeit.timeit(test1, setup=setup, number=100000))
print("Without import:", timeit.timeit(test2, setup=setup, number=100000))
