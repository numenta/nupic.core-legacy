# TODO: SOURCE, COPYRIGHT?, LICENSE?
import numpy as np


# Matrix for converting axial coordinates to pixel coordinates
axial_to_pixel_mat = np.array([[np.sqrt(3), np.sqrt(3) / 2], [0, 3 / 2.]])

# Matrix for converting pixel coordinates to axial coordinates
pixel_to_axial_mat = np.linalg.inv(axial_to_pixel_mat)


# These are the vectors for moving from any hex to one of its neighbors.
SE = np.array((1, 0, -1))
SW = np.array((0, 1, -1))
W = np.array((-1, 1, 0))
NW = np.array((-1, 0, 1))
NE = np.array((0, -1, 1))
E = np.array((1, -1, 0))
ALL_DIRECTIONS = np.array([NW, NE, E, SE, SW, W, ])


def get_cube_distance(hex_start, hex_end):
    """
    Computes the smallest number of hexes between hex_start and hex_end, on the hex lattice.
    :param hex_start: Starting hex...
    :param hex_end: Ending hex...
    :return: Smallest number of hexes between `hex_start` and `hex_end`, on hex lattice.
    """
    return np.sum(np.abs(hex_start - hex_end) / 2)


# Selection Functions ######

def get_neighbor(hex, direction):
    """
    Simply returns the neighbor, in the direction specified, of the hexagon.
    :param hex: Cube coordinates of the hexagon.
    :param direction: A direction from the DIR class.
    :return: The location of the neighbor in cube coordinates.
    """
    return hex + direction


def get_ring(center, radius):
    """
    Retrieves the locations of all the hexes exactly a certain distance from a hexagon.
    :param center: The location of the hexagon to get the ring of.
    :param radius: The distance from `center` of the hexes we want.
    :return: An array of locations of the hexes that are exactly `radius` units away from `center`.
    """
    if radius < 0:
        return []
    if radius == 0:
        return [center]

    rad_hex = np.zeros((6 * radius, 3))
    count = 0
    for i in range(0, 6):
        for k in range(0, radius):
            rad_hex[count] = ALL_DIRECTIONS[i - 1] * (radius - k) + ALL_DIRECTIONS[i] * (k)
            count += 1

    return np.squeeze(rad_hex) + center


def get_disk(center, radius):
    """
    Retrieves the locations of all the hexes within certain distance from a hexagon.
    :param center: The location of the hexagon to get the neighbors of.
    :param radius: The distance from `center` of the hexes we want.
    :return: An array of locations of the hexes that are within `radius` units away from `center`.
    """
    hex_set = []
    for x in range(-radius, radius + 1):
        for y in range(max(-radius, -(x + radius)), min(radius + 1, -x + radius + 1)):
            z = -x - y
            hex_set.append(np.array([x, y, z]))
    return np.array(hex_set) + center


def get_spiral(center, radius_start=1, radius_end=2):
    """
    Retrieves all hexes that are `radius` hexes away from the `center`.
    :param center: The location of the center hex.
    :param radius_start: The distance from center. We want all hexes greater than or equal to this distance.
    :param radius_end: The distance from center. We want all hexes within this distance from `center`.
    :return: An array of locations of the hexes that are within `radius` hexes away from `center`.
    """
    hex_area = get_ring(center, 0)
    for i in range(radius_start, radius_end + 1):
        hex_area = np.append(hex_area, get_ring(center, i), axis=0)
    return hex_area


def get_hex_line(hex_start, hex_end):
    """
    Get hexes on line from hex_start to hex_end.
    :param hex_start: The hex where the line starts.
    :param hex_end: The hex where the line ends.
    :return: A set of hexes along a straight line from hex_start to hex_end.
    """
    hex_distance = get_cube_distance(hex_start, hex_end)
    if hex_distance < 1:
        return np.array([hex_start])

    # Set up linear system to compute linearly interpolated cube points
    bottom_row = np.array([i / hex_distance for i in np.arange(hex_distance)])
    x = np.vstack((1 - bottom_row, bottom_row))
    A = np.vstack((hex_start, hex_end)).T

    # linearly interpolate from a to b in n steps
    interpolated_points = A.dot(x)
    interpolated_points = np.vstack((interpolated_points.T, hex_end))
    return np.array(cube_round(interpolated_points))


# Conversion Functions ######

def cube_to_axial(cube):
    """
    Convert cube to axial coordinates.
    :param cube: A coordinate in cube form. nx3
    :return: `cube` in axial form.
    """
    return np.vstack((cube[:, 0], cube[:, 2])).T


def axial_to_cube(axial):
    """
    Convert axial to cube coordinates.
    :param axial: A coordinate in axial form.
    :return: `axial` in cube form.
    """
    x = axial[:, 0]
    z = axial[:, 1]
    y = -x - z
    cube_coords = np.vstack((x, y, z)).T
    return cube_coords


def axial_to_pixel(axial, radius):
    """
    Converts the location of a hex in axial form to pixel coordinates.
    :param axial: The location of a hex in axial form. nx3
    :param radius: Radius of all hexagons.
    :return: `axial` in pixel coordinates.
    """
    pos = radius * axial_to_pixel_mat.dot(axial.T)
    return pos.T


def cube_to_pixel(cube, radius):
    """
    Converts the location of a hex in cube form to pixel coordinates.
    :param cube: The location of a hex in cube form. nx3
    :param radius: Radius of all hexagons.
    :return: `cube` in pixel coordinates.
    """
    in_axial_form = cube_to_axial(cube)
    return axial_to_pixel(in_axial_form, radius)


def pixel_to_cube(pixel, radius):
    """
    Converts the location of a hex in pixel coordinates to cube form.
    :param pixel: The location of a hex in pixel coordinates. nx2
    :param radius: Radius of all hexagons.
    :return: `pixel` in cube coordinates.
    """
    axial = pixel_to_axial_mat.dot(pixel.T) / radius
    return cube_round(axial_to_cube(axial.T))


def pixel_to_axial(pixel, radius):
    """
    Converts the location of a hex in pixel coordinates to axial form.
    :param pixel: The location of a hex in pixel coordinates. nx2
    :param radius: Radius of all hexagons.
    :return: `pixel` in axial coordinates.
    """
    cube = pixel_to_cube(pixel, radius)
    return cube_to_axial(cube)


def cube_round(cubes):
    """
    Rounds a location in cube coordinates to the center of the nearest hex.
    :param cubes: Locations in cube form. nx3
    :return: The location of the center of the nearest hex in cube coordinates.
    """
    rounded = np.zeros((cubes.shape[0], 3))
    rounded_cubes = np.round(cubes)
    for i, cube in enumerate(rounded_cubes):
        (rx, ry, rz) = cube
        xdiff, ydiff, zdiff = np.abs(cube-cubes[i])
        if xdiff > ydiff and xdiff > zdiff:
            rx = -ry - rz
        elif ydiff > zdiff:
            ry = -rx - rz
        else:
            rz = -rx - ry
        rounded[i] = (rx, ry, rz)
    return rounded


def axial_round(axial):
    """
    Rounds a location in axial coordinates to the center of the nearest hex.
    :param axial: A location in axial form. nx2
    :return: The location of the center of the nearest hex in axial coordinates.
    """
    return cube_to_axial(cube_round(axial_to_cube(axial)))
