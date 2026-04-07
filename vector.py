"""
vector.py
---------
Mmodule for 3D vector geometry.

Provides:
  - Vector : a 3D vector defined by two points, with an orthonormal
             basis (e1, e2, e3) and a length method.
  - segment_distance : the perpendicular distance from any point P to
                       a line segment defined by two endpoints A and B.

Intended use
------------
This module is fully general — A, B, and P can be any 3D positions,
whether galaxy coordinates, particle positions, etc. 

Note on origin
--------------
This module was originally written to compute distances from galaxy
positions to cosmic filament segments in a large-scale structure
analysis. In that context, A and B are three dimensional filament node positions and
P is a galaxy position. The module has been written generically so
that it can be reused for any analogous point-to-segment distance
problem in 3D. I originally had it work with only galaxies but realized it's 
more useful to specify that it can be used for more. 

The segment_distance function uses
the Vector class to:
  1. Check whether P projects onto the segment AB or is closest to an
     endpoint, using dot product conditions.
  2. If P projects onto the segment, construct the (e2, e3) plane
     perpendicular to the segment at A and return the Euclidean length
     of the projection of AP onto that plane — i.e. the perpendicular
     distance from P to the infinite line through AB, restricted to the
     segment.

Example usage
-------------
    from vector import Vector, segment_distance
    import numpy as np

    A = np.array([0.0, 0.0, 0.0])   # start of segment
    B = np.array([1.0, 0.0, 0.0])   # end of segment
    P = np.array([0.5, 1.0, 0.0])   # query point

    d = segment_distance(A, B, P)   # returns 1.0

Dependencies
------------
    numpy
"""

import numpy as np


# =============================================================================
# Vector class
# =============================================================================

class Vector:
    """
    A 3D vector defined by two points A (start) and B (end),
    representing the displacement from A to B.

    In the context of segment distance calculations:
      - A and B are the two endpoints of a line segment
      - The Vector can also be constructed from a segment endpoint
        to a query point P (e.g. Vector(A, P) gives the vector AP)

    Methods
    -------
    length()
        Euclidean magnitude of the vector AB.
    e1()
        Unit vector along AB — the axis of the segment.
    e2()
        Unit vector perpendicular to e1, constructed via cross product
        with an arbitrary non-parallel vector. Together with e3, spans
        the plane perpendicular to the segment.
    e3()
        Unit vector perpendicular to both e1 and e2, completing the
        right-handed orthonormal basis (e1, e2, e3).

    Attributes
    ----------
    A : np.ndarray, shape (3,)
        Start point of the vector.
    B : np.ndarray, shape (3,)
        End point of the vector.
    _vec : np.ndarray, shape (3,)
        Raw displacement array B - A. Accessed directly in dot product
        calculations for efficiency.

    Example
    -------
        A = np.array([0., 0., 0.])
        B = np.array([1., 0., 0.])
        v = Vector(A, B)
        v.length()   # 1.0
        v.e1()       # array([1., 0., 0.])
        v.e2()       # some unit vector perpendicular to e1
        v.e3()       # unit vector perpendicular to both e1 and e2
    """

    def __init__(self, A, B):
        """
        Parameters
        ----------
        A : array-like of shape (3,)
            Start point of the vector. In segment calculations this is
            typically one endpoint of the segment, or an endpoint and
            a query point (e.g. A and P to form vector AP).
        B : array-like of shape (3,)
            End point of the vector. In segment calculations this is
            typically the other endpoint, or the query point.
        """
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        # Raw displacement vector — stored once and reused by all methods
        self._vec = self.B - self.A

    def length(self):
        """
        Returns the Euclidean magnitude of the vector AB, i.e. |B - A|.
        """
        return np.linalg.norm(self._vec)

    def e1(self):
        """
        Returns the unit vector along AB — the direction of the segment.

        This forms the first axis of the orthonormal basis. In segment
        distance calculations, e1 points along the filament/segment axis.

        Raises
        ------
        ValueError
            If A and B are the same point (zero-length vector), since
            no direction can be defined.
        """
        l = self.length()
        if l == 0:
            raise ValueError("Cannot define e1: A and B are the same point.")
        return self._vec / l

    def e2(self):
        """
        Returns a unit vector perpendicular to e1.

        Constructed by taking the cross product of e1 with an arbitrary
        vector that is guaranteed not to be parallel to e1. Together with
        e3, this spans the plane perpendicular to the segment — i.e. the
        plane onto which a query point P is projected to measure its
        distance from the segment axis.

        The choice of e2 direction within the perpendicular plane is
        arbitrary, but this does not affect distance calculations since
        the distance is computed as sqrt(proj_e2^2 + proj_e3^2), which
        is invariant to rotation within the plane.
        """
        e1_vec = self.e1()

        # Choose an arbitrary vector not parallel to e1.
        # The dot product check detects near-parallelism (|cos θ| > 0.9),
        # which would cause the cross product to be near-zero and unstable.
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(e1_vec, arbitrary)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])

        # Cross product guarantees perpendicularity to e1 by definition
        e2 = np.cross(e1_vec, arbitrary)
        return e2 / np.linalg.norm(e2)

    def e3(self):
        """
        Returns a unit vector perpendicular to both e1 and e2, completing
        the right-handed orthonormal basis (e1, e2, e3).

        Together with e2, e3 spans the plane perpendicular to the segment.
        Any point P's displacement from the segment axis can be fully
        described by its projections onto e2 and e3.
        """
        e3 = np.cross(self.e1(), self.e2())
        return e3 / np.linalg.norm(e3)


# =============================================================================
# Segment distance function
# =============================================================================

def segment_distance(A, B, P):
    """
    Compute the minimum distance from a point P to a line segment AB.

    This function determines whether P is closest to the interior of
    the segment or to one of its endpoints, using dot product conditions.
    If P projects onto the segment interior, the perpendicular distance
    is computed by projecting the vector AP onto the (e2, e3) plane
    perpendicular to AB.

    This is fully general — A, B, and P can be any 3D positions
    (e.g. filament nodes and galaxy positions, mesh vertices and a
    probe point, path waypoints and a query location, etc.)

    Parameters
    ----------
    A : array-like of shape (3,)
        Start endpoint of the segment. In the original use case this
        is a filament node position in physical Mpc.
    B : array-like of shape (3,)
        End endpoint of the segment. In the original use case this
        is the next filament node position along the filament.
    P : array-like of shape (3,)
        The query point whose distance to segment AB is sought.
        In the original use case this is a galaxy position in physical
        Mpc, but can be any 3D point.

    Returns
    -------
    float
        Minimum distance from P to the closest point on segment AB.

    Notes
    -----
    The three cases are determined by dot product conditions:

      AB · AP < 0  →  P is "behind" A (outside the segment on A's side)
                      Closest point on segment is A. Return |AP|.

      BA · BP < 0  →  P is "beyond" B (outside the segment on B's side)
                      Closest point on segment is B. Return |BP|.

      Otherwise    →  P projects onto the segment interior.
                      Build the (e2, e3) plane perpendicular to AB at A,
                      project AP onto it, return sqrt(proj_e2^2 + proj_e3^2).

    Example
    -------
        A = np.array([0., 0., 0.])
        B = np.array([1., 0., 0.])
        P = np.array([0.5, 1.0, 0.])
        segment_distance(A, B, P)   # returns 1.0

        P2 = np.array([-1., 1., 0.])
        segment_distance(A, B, P2)  # returns sqrt(2) ≈ 1.414, closest to A
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    P = np.asarray(P, dtype=float)

    # Build vectors from segment endpoints to query point
    AB = Vector(A, B)   # segment direction
    AP = Vector(A, P)   # from segment start to query point
    BA = Vector(B, A)   # reverse segment direction
    BP = Vector(B, P)   # from segment end to query point

    # Case 1: P is behind A — closest point on segment is A
    if np.dot(AB._vec, AP._vec) < 0:
        return AP.length()

    # Case 2: P is beyond B — closest point on segment is B
    if np.dot(BA._vec, BP._vec) < 0:
        return BP.length()

    # Case 3: P projects onto the segment interior.
    # Project AP onto the (e2, e3) plane perpendicular to AB.
    # The length of this projection is the perpendicular distance
    # from P to the infinite line through AB, which equals the
    # distance to the segment since P projects onto its interior.
    e2 = AB.e2()
    e3 = AB.e3()

    proj_e2 = np.dot(AP._vec, e2)
    proj_e3 = np.dot(AP._vec, e3)

    return np.sqrt(proj_e2**2 + proj_e3**2)