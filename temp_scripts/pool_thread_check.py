from multiprocessing.dummy import Pool as ThreadPool
import os
import threading
from enum import  Enum

class Enum_Test(Enum):
    e1 = 3
    e2 = 4




def squareNumber(n):
    print threading.current_thread().ident
    return n ** 2

# function to be mapped over
def calculateParallel(numbers, threads=2):
    pool = ThreadPool(threads)
    results = pool.map(squareNumber, numbers)
    pool.close()
    pool.join()
    return results

if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]
    squaredNumbers = calculateParallel(numbers, 4)
    for n in squaredNumbers:
        print(n)

    d = {}
    d[int(Enum_Test.e1)] = 44

    print d[44]