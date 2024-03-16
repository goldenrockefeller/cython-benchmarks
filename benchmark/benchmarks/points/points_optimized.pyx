"""
Artificial, floating point-heavy benchmark originally used by Factor.

Adapted to mypyc by Jukka Lehtosalo
"""
from __future__ import annotations
from libc.math cimport sin, cos, sqrt # 11x boost
cimport cython

from benchmarking import benchmark


POINTS = 100000

# create_point give 5x boost
cdef Point create_point(double i):
    cdef Point p = Point.__new__(Point)
    initiate_point(p, i)
    return p

cpdef void initiate_point(Point self, double i):
    cdef double x
    self.x = x = sin(i)
    self.y = cos(i) * 3
    self.z = (x * x) / 2

cdef class Point():
    cdef public double x
    cdef public double y
    cdef public double z

    def __init__(self, i: float) -> None:
        initiate_point(self, i)

    def __repr__(self) -> str:
        return "<Point: x=%s, y=%s, z=%s>" % (self.x, self.y, self.z)

    cdef void normalize(self):
        x = self.x
        y = self.y
        z = self.z
        norm = sqrt(x * x + y * y + z * z)
        self.x /= norm
        self.y /= norm
        self.z /= norm

    cdef Point maximize(self, Point other):
        self.x = self.x if self.x > other.x else other.x
        self.y = self.y if self.y > other.y else other.y
        self.z = self.z if self.z > other.z else other.z
        return self


cpdef Point maximize(list points):
    cdef Point next = points[0]
    for p in points: # 1x effect
        next = next.maximize(p)
    return next

cpdef bm_float():
    cdef Py_ssize_t n = POINTS
    cdef points = [] #no effect
    cdef Point p
    cdef Py_ssize_t i # 2x effect
    for i in range(n):
        points.append(create_point(i))
    for p in points:
        p.normalize()
    cdef Point result = maximize(points)
    str(result)