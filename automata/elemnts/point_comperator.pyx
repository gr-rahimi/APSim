from itertools import izip
def cython_can_accept(left, point, right):
    for l,p,r in izip(left, point, right):
        if l <= p <= r:
            continue
        else:
            return False
    return True


