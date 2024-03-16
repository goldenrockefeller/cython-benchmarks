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
    p.direct = 2
    initiate_point(p, i)
    return p

cpdef void initiate_point(Point self, double i):
    cdef double x
    self.x = x = sin(i)
    self.y = cos(i) * 3
    self.z = (x * x) / 2

cdef inline void set_x(Point self, double val) noexcept:
    if self.direct:
        self.x = val
    else:
        self.set_x(val)

cdef inline void set_y(Point self, double val) noexcept:
    if self.direct:
        self.y = val
    else:
        self.set_y(val)

cdef inline void set_z(Point self, double val) noexcept:
    if self.direct:
        self.z = val
    else:
        self.set_z(val)

cdef inline double get_x(Point self) noexcept:
    if self.direct:
        return self.x
    else:
        return self.get_x()

cdef inline double get_y(Point self) noexcept:
    if self.direct:
        return self.y
    else:
        return self.get_y()

cdef inline double get_z(Point self):
    if self.direct:
        return self.z
    else:
        return self.get_z()

cdef class Point():
    cdef char direct
    cdef public double x
    cdef public double y
    cdef public double z

    def __init__(self, i: float) -> None:
        initiate_point(self, i)

    def __repr__(self) -> str:
        return "<Point: x=%s, y=%s, z=%s>" % (self.x, self.y, self.z)

    cdef void set_x(self, double val):
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
        x = get_x(self)
        y = get_y(self)
        z = get_z(self)
        norm = sqrt(x * x + y * y + z * z)
        set_x(self, x / norm)
        set_y(self, y / norm)
        set_z(self, z / norm)

    cdef Point maximize(self, Point other):
        set_x(self, get_x(self) if get_x(self) > get_x(self) else get_x(self))
        set_y(self, get_y(self) if get_y(self) > get_y(self) else get_y(self))
        set_z(self, get_z(self) if get_z(self) > get_z(self) else get_z(self))
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