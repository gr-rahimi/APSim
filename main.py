import enum

class Test(enum.Enum):
    A = 6
    B = 2


for e in Test:
    print e
