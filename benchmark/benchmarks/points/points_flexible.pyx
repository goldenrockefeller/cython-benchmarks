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

    cdef void set_x(self, double val): # noexcept give 2x boost
        self.x = val

    cdef void set_y(self, double val):
        self.y = val

    cdef void set_z(self, double val):
        self.z = val

    cdef double get_x(self):
        return self.x

    cdef double get_y(self):
        return self.y

    cdef double get_z(self):
        return self.z

    cdef void normalize(self):
        x = self.get_x()
        y = self.get_y()
        z = self.get_z()
        norm = sqrt(x * x + y * y + z * z)
        self.set_x(x / norm)
        self.set_y(y / norm)
        self.set_z(z / norm)

    cdef Point maximize(self, Point other):
        self.set_x(self.get_x() if self.get_x() > other.get_x() else other.get_x())
        self.set_y(self.get_y() if self.get_y() > other.get_y() else other.get_y())
        self.set_z(self.get_z() if self.get_z() > other.get_z() else other.get_z())
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