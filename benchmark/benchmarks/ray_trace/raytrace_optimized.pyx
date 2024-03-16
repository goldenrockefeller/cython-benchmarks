# mypy: disallow-untyped-defs

"""
This file contains definitions for a simple raytracer.
Copyright Callum and Tony Garnock-Jones, 2008.

This file may be freely redistributed under the MIT license,
http://www.opensource.org/licenses/mit-license.php

From http://www.lshift.net/blog/2008/10/29/toy-raytracer-in-python

Migrated to mypyc by Jukka Lehtosalo.
"""

from __future__ import annotations

import array
cimport libc.math as math

from typing import Tuple, overload

from typing import Tuple, overload, Final

from benchmarking import benchmark


DEFAULT_WIDTH: Final = 100
DEFAULT_HEIGHT: Final = 100
EPSILON: Final = 0.00001

cdef Vector create_vector(initx: float, inity: float, initz: float):
    cdef Vector vector = Vector.__new__(Vector)
    initialize_vector(vector, initx, inity, initz)
    return vector

cdef void initialize_vector(self: Vector, initx: float, inity: float, initz: float):
    self.x = initx
    self.y = inity
    self.z = initz

cdef class Vector:
    cdef public double x
    cdef public double y
    cdef public double z

    def __init__(self, initx: float, inity: float, initz: float) -> None:
        raise RuntimeError
        self.x = initx
        self.y = inity
        self.z = initz

    def __str__(self) -> str:
        return '(%s,%s,%s)' % (self.x, self.y, self.z)

    def __repr__(self) -> str:
        return 'Vector(%s,%s,%s)' % (self.x, self.y, self.z)

    def magnitude(self) -> float:
        return math.sqrt(self.dot(self))

    @overload
    def __add__(self, other: Vector) -> Vector: ...
    @overload
    def __add__(self, other: Point) -> Point: ...
    def __add__(self, other: Vector | Point) -> Vector | Point:
        raise RuntimeError
        if other.isPoint():
            return Point(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    cdef Vector add_vector(self, other: Vector):
        return create_vector(self.x + other.x, self.y + other.y, self.z + other.z)

    cdef Point add_point(self, other: Point):
        return create_point(self.x + other.x, self.y + other.y, self.z + other.z)

    cdef Vector sub(self, other: Vector):
        return create_vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __sub__(self, other: Vector) -> Vector:
        raise RuntimeError
        other.mustBeVector()
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    cdef Vector scale(self, factor: float):
        return create_vector(factor * self.x, factor * self.y, factor * self.z)

    cdef double dot(self, other: Vector):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    cdef Vector cross(self, other: Vector):
        return create_vector(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)

    cdef Vector normalized(self):
        return self.scale(1.0 / self.magnitude())

    cdef Vector negated(self):
        return self.scale(-1)

    def __eq__(self, other: object) -> bool:
        raise RuntimeError
        if not isinstance(other, Vector):
            return False
        return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)

    cdef bint eq(self, other):
        if not isinstance(other, Vector):
            return False
        cdef Vector vector = other
        return  (self.x == other.x) and (self.y == other.y) and (self.z == other.z)


    cdef Vector reflectThrough(self, normal: Vector):
        d = normal.scale(self.dot(normal)) #Should be inferred as Vector CHECK
        return self .sub( d.scale(2))


cdef Vector Vector_ZERO = create_vector(0, 0, 0)
cdef Vector Vector_RIGHT = create_vector(1, 0, 0)
cdef Vector Vector_UP = create_vector(0, 1, 0)
cdef Vector Vector_OUT = create_vector(0, 0, 1)

cdef Point create_point(initx: float, inity: float, initz: float):
    cdef Point point = Point.__new__(Point)
    initialize_point(point, initx, inity, initz)
    return point

cdef void initialize_point(self: Point, initx: float, inity: float, initz: float):
    self.x = initx
    self.y = inity
    self.z = initz

cdef class Point:

    cdef public double x
    cdef public double y
    cdef public double z

    def __init__(self, initx: float, inity: float, initz: float) -> None:
        self.x = initx
        self.y = inity
        self.z = initz

    def __str__(self) -> str:
        return '(%s,%s,%s)' % (self.x, self.y, self.z)

    def __repr__(self) -> str:
        return 'Point(%s,%s,%s)' % (self.x, self.y, self.z)

    cdef Point add_point(self, other: Point):
        return create_point(self.x + other.x, self.y + other.y, self.z + other.z)

    @overload
    def __sub__(self, other: Vector) -> Point: ...
    @overload
    def __sub__(self, other: Point) -> Vector: ...
    def __sub__(self, other: Point | Vector) -> Point | Vector:
        raise RuntimeError
        if isinstance(other, Point):
            return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    cdef Point sub_vector(self, other: Vector):
        return create_point(self.x - other.x, self.y - other.y, self.z - other.z)

    cdef Vector sub_point(self, other: Point):
        return create_vector(self.x - other.x, self.y - other.y, self.z - other.z)

cdef class Object:
    cpdef intersectionTime(self, ray: Ray):
        raise NotImplementedError

    cpdef Vector normalAt(self, p: Point):
        raise NotImplementedError

cdef Sphere create_sphere(centre: Point, radius: float):
    cdef Sphere sphere = Sphere.__new__(Sphere)
    initialize_sphere(sphere, centre, radius)
    return sphere

cdef void initialize_sphere(self: Sphere, centre: Point, radius: float):
    self.centre = centre
    self.radius = radius

cdef class Sphere(Object):

    cdef public Point centre
    cdef public float radius

    def __init__(self, centre: Point, radius: float) -> None:
        raise RuntimeError
        self.centre = centre
        self.radius = radius

    def __repr__(self) -> str:
        return 'Sphere(%s,%s)' % (repr(self.centre), self.radius)

    cpdef intersectionTime(self, ray: Ray):
        cp = self.centre.sub_point(ray.point)
        v = ray.vector.dot(cp)
        discriminant = (self.radius * self.radius) - (cp.dot(cp) - v * v)
        if discriminant < 0:
            return None
        else:
            return v - math.sqrt(discriminant)

    cpdef Vector normalAt(self, p: Point):
        return (p .sub_point(self.centre)).normalized()

cdef Halfspace create_halfspace(point: Point, normal: Vector):
    cdef Halfspace halfspace = Halfspace.__new__(Halfspace)
    initialize_halfspace(halfspace, point, normal)
    return halfspace

cdef void initialize_halfspace(self: Halfspace, point: Point, normal: Vector):
    self.point = point
    self.normal = normal.normalized()

cdef class Halfspace(Object):
    cdef public Point point
    cdef public Vector normal

    def __init__(self, point: Point, normal: Vector) -> None:
        raise RuntimeError
        self.point = point
        self.normal = normal.normalized()

    def __repr__(self) -> str:
        return 'Halfspace(%s,%s)' % (repr(self.point), repr(self.normal))

    cpdef intersectionTime(self, ray: Ray):
        v = ray.vector.dot(self.normal)
        if v:
            return 1 / -v
        else:
            return None

    cpdef Vector normalAt(self, p: Point):
        return self.normal

cdef Ray create_ray(point: Point, vector: Vector):
    cdef Ray ray = Ray.__new__(Ray)
    initialize_ray(ray, point, vector)
    return ray

cdef void initialize_ray(self: Ray, point: Point, vector: Vector):
    self.point = point
    self.vector = vector.normalized()

cdef class Ray:
    cdef public Point point
    cdef public Vector vector

    def __init__(self, point: Point, vector: Vector) -> None:
        raise RuntimeError
        self.point = point
        self.vector = vector.normalized()

    def __repr__(self) -> str:
        return 'Ray(%s,%s)' % (repr(self.point), repr(self.vector))

    cdef Point pointAtTime(self, t: float):
        return self.vector.scale(t).add_point(self.point)


cdef Point Point_ZERO = Point(0, 0, 0)

cdef Canvas create_canvas(width: int, height: int):
    cdef Canvas canvas = Canvas.__new__(Canvas)
    initialize_canvas(canvas, width, height)
    return canvas

cdef void initialize_canvas(self: Canvas, width: int, height: int):
    self.bytes = array.array('B', [0] * (width * height * 3))
    for i in range(width * height):
        self.bytes[i * 3 + 2] = 255
    self.width = width
    self.height = height


cdef class Canvas:
    cdef public unsigned char[:] bytes
    cdef public Py_ssize_t width
    cdef public Py_ssize_t height

    def __init__(self, width: int, height: int) -> None:
        self.bytes = array.array('B', [0] * (width * height * 3))
        for i in range(width * height): # CHECK i is integer
            self.bytes[i * 3 + 2] = 255
        self.width = width
        self.height = height

    cdef void plot(self, x: Py_ssize_t, y: Py_ssize_t, r: float, g: float, b: float) :
        i = ((self.height - y - 1) * self.width + x) * 3
        self.bytes[i] = max(0, min(255, int(r * 255)))
        self.bytes[i + 1] = max(0, min(255, int(g * 255)))
        self.bytes[i + 2] = max(0, min(255, int(b * 255)))

    cdef void write_ppm(self, filename: str) :
        header = 'P6 %d %d 255\n' % (self.width, self.height)
        with open(filename, "wb") as fp:
            fp.write(header.encode('ascii'))
            fp.write(self.bytes.tobytes())


def firstIntersection(
    intersections: list[tuple[Object, float | None, Surface]]
) -> tuple[Object, float, Surface] | None:
    result: tuple[Object, float, Surface] | None = None
    for i in intersections:
        candidateT = i[1]
        if candidateT is not None and candidateT > -EPSILON:
            if result is None or candidateT < result[1]:
                result = (i[0], candidateT, i[2])
    return result


Colour = Tuple[float, float, float]


cdef Scene create_scene():
    cdef Scene scene = Scene.__new__(Scene)
    initialize_scene(scene)
    return scene

cdef void initialize_scene(self: Scene):
    self.objects: list[tuple[Object, Surface]] = []
    self.lightPoints: list[Point] = []
    self.position = Point(0, 1.8, 10)
    self.lookingAt = Point_ZERO
    self.fieldOfView: float = 45.0
    self.recursionDepth = 0


cdef class Scene:
    cdef public list objects
    cdef public list lightPoints
    cdef public Point position
    cdef public Point lookingAt
    cdef public double fieldOfView
    cdef public double recursionDepth

    def __init__(self) -> None:
        self.objects: list[tuple[Object, Surface]] = []
        self.lightPoints: list[Point] = []
        self.position = Point(0, 1.8, 10)
        self.lookingAt = Point_ZERO
        self.fieldOfView: float = 45.0
        self.recursionDepth = 0

    cdef void moveTo(self, p: Point):
        self.position = p

    cdef void lookAt(self, p: Point):
        self.lookingAt = p

    cdef addObject(self, object: Object, surface: Surface):
        self.objects.append((object, surface))

    cdef addLight(self, p: Point):
        self.lightPoints.append(p)

    cdef render(self, canvas: Canvas):

        cdef Py_ssize_t y
        cdef Py_ssize_t x

        # Should be inference CHECK
        cdef double fovRadians = math.pi * (self.fieldOfView / 2.0) / 180.0
        halfWidth = math.tan(fovRadians)
        halfHeight = 0.75 * halfWidth
        width = halfWidth * 2
        height = halfHeight * 2
        pixelWidth = width / (canvas.width - 1)
        pixelHeight = height / (canvas.height - 1)

        eye = create_ray(self.position, self.lookingAt .sub_point( self.position))
        vpRight = eye.vector.cross(Vector_UP).normalized()
        vpUp = vpRight.cross(eye.vector).normalized()

        for y in range(canvas.height):
            for x in range(canvas.width):
                xcomp = vpRight.scale(x * pixelWidth - halfWidth)
                ycomp = vpUp.scale(y * pixelHeight - halfHeight)
                ray = create_ray(eye.point, eye.vector.add_vector( xcomp).add_vector(ycomp))
                colour = self.rayColour(ray)
                canvas.plot(x, y, colour[0],  colour[1],  colour[2])

    cdef tuple rayColour(self, ray: Ray):
        cdef Object o
        cdef Surface s
        if self.recursionDepth > 3:
            return (0, 0, 0)
        try:
            self.recursionDepth = self.recursionDepth + 1
            intersections = [(o, o.intersectionTime(ray), s)
                             for (o, s) in self.objects]
            i = firstIntersection(intersections)
            if i is None:
                return (0, 0, 0)  # the background colour
            else:
                (o, t, s) = i
                p = ray.pointAtTime(t)
                return s.colourAt(self, ray, p, o.normalAt(p))
        finally:
            self.recursionDepth = self.recursionDepth - 1

    cdef bint _lightIsVisible(self, l: Point, p: Point) :
        cdef Object o
        cdef Surface s
        for (o, s) in self.objects:
            t = o.intersectionTime(create_ray(p, l .sub_point(p)))
            if t is not None and t > EPSILON:
                return False
        return True

    cdef list visibleLights(self, p: Point):
        result = []
        for l in self.lightPoints:
            if self._lightIsVisible(l, p):
                result.append(l)
        return result

cdef tuple addColours(a: tuple, scale: float, b: tuple):
    return (<float>a[0] + scale * <float>b[0],
           <float> a[1] + scale * <float>b[1],
            <float>a[2] + scale * <float>b[2])

cdef class Surface:
    cdef tuple colourAt(self, scene: Scene, ray: Ray, p: Point, normal: Vector):
        raise NotImplementedError

cdef SimpleSurface create_simple_surface(
                 baseColour: tuple = (1, 1, 1),
                 specularCoefficient: float = 0.2,
                 lambertCoefficient: float = 0.6):
    cdef SimpleSurface simple_surface = SimpleSurface.__new__(SimpleSurface)
    initialize_simple_surface(simple_surface, baseColour, specularCoefficient, lambertCoefficient)
    return simple_surface

cdef void initialize_simple_surface(self: SimpleSurface,
                 baseColour: Colour = (1, 1, 1),
                 specularCoefficient: float = 0.2,
                 lambertCoefficient: float = 0.6):

    self.baseColour = baseColour
    self.specularCoefficient = specularCoefficient
    self.lambertCoefficient = lambertCoefficient
    self.ambientCoefficient = 1.0 - self.specularCoefficient - self.lambertCoefficient

cdef class SimpleSurface(Surface):
    cdef public tuple baseColour
    cdef public float specularCoefficient
    cdef public float lambertCoefficient
    cdef public float ambientCoefficient

    def __init__(self,
                 *,
                 baseColour: Colour = (1, 1, 1),
                 specularCoefficient: float = 0.2,
                 lambertCoefficient: float = 0.6) -> None:
        self.baseColour = baseColour
        self.specularCoefficient = specularCoefficient
        self.lambertCoefficient = lambertCoefficient
        self.ambientCoefficient = 1.0 - self.specularCoefficient - self.lambertCoefficient

    cdef tuple baseColourAt(self, p: Point):
        return self.baseColour

    cdef tuple colourAt(self, scene: Scene, ray: Ray, p: Point, normal: Vector):
        cdef tuple c
        b = self.baseColourAt(p)
        cdef Point lightPoint

        c: Colour = (0, 0, 0)
        if self.specularCoefficient > 0:
            reflectedRay = create_ray(p, ray.vector.reflectThrough(normal))
            reflectedColour = scene.rayColour(reflectedRay)
            c = addColours(c, self.specularCoefficient, reflectedColour)

        if self.lambertCoefficient > 0:
            lambertAmount: float = 0.0
            for lightPoint in scene.visibleLights(p): # CHECK light point is of type Point
                contribution = (lightPoint .sub_point(p)).normalized().dot(normal)
                if contribution > 0:
                    lambertAmount = lambertAmount + contribution
            lambertAmount = min(1, lambertAmount)
            c = addColours(c, self.lambertCoefficient * lambertAmount, b)

        if self.ambientCoefficient > 0:
            c = addColours(c, self.ambientCoefficient, b)

        return c


cdef CheckerboardSurface create_checkerboard_surface(
                 baseColour: Colour = (1, 1, 1),
                 specularCoefficient: float = 0.2,
                 lambertCoefficient: float = 0.6,
                 otherColour: Colour = (0, 0, 0),
                 checkSize: float = 1):
    cdef CheckerboardSurface checkerboard_surface = CheckerboardSurface.__new__(CheckerboardSurface)
    initialize_checkerboard_surface(checkerboard_surface, baseColour, specularCoefficient, lambertCoefficient, otherColour, checkSize)
    return checkerboard_surface

cdef void initialize_checkerboard_surface(self: CheckerboardSurface,
                 baseColour: Colour = (1, 1, 1),
                 specularCoefficient: float = 0.2,
                 lambertCoefficient: float = 0.6,
                 otherColour: Colour = (0, 0, 0),
                 checkSize: float = 1):
    initialize_simple_surface(self, baseColour, specularCoefficient, lambertCoefficient)
    self.otherColour = otherColour
    self.checkSize = checkSize

cdef class CheckerboardSurface(SimpleSurface):

    cdef public tuple otherColour
    cdef public float checkSize

    def __init__(self,
                 *,
                 baseColour: Colour = (1, 1, 1),
                 specularCoefficient: float = 0.2,
                 lambertCoefficient: float = 0.6,
                 otherColour: Colour = (0, 0, 0),
                 checkSize: float = 1) -> None:
        super().__init__(baseColour=baseColour,
                         specularCoefficient=specularCoefficient,
                         lambertCoefficient=lambertCoefficient)
        self.otherColour = otherColour
        self.checkSize = checkSize

    cdef tuple baseColourAt(self, p: Point):
        v = p .sub_point( Point_ZERO)
        v.scale(1.0 / self.checkSize)
        if ((int(abs(v.x) + 0.5)
             + int(abs(v.y) + 0.5)
             + int(abs(v.z) + 0.5)) % 2):
            return self.otherColour
        else:
            return self.baseColour


cpdef void bench_raytrace(loops: int, width: int, height: int, filename: str | None):
    range_it = range(loops)

    for i in range_it:
        canvas = create_canvas(width, height)
        s = create_scene()
        s.addLight(create_point(30, 30, 10))
        s.addLight(create_point(-10, 100, 30))
        s.lookAt(create_point(0, 3, 0))
        s.addObject(create_sphere(create_point(1, 3, -10), 2),
                    create_simple_surface(baseColour=(1, 1, 0)))
        for y in range(6):
            s.addObject(create_sphere(create_point(-3 - y * 0.4, 2.3, -5), 0.4),
                        create_simple_surface(baseColour=(y / 6.0, 1 - y / 6.0, 0.5)))
        s.addObject(create_halfspace(create_point(0, 0, 0), Vector_UP),
                    create_checkerboard_surface())
        s.render(canvas)

    if filename:
        canvas.write_ppm(filename)

def bm_raytrace() -> None:
    bench_raytrace(1, DEFAULT_WIDTH, DEFAULT_HEIGHT, None)