import numpy as np


def change_name_for_unix(string):
    """Adapt a string to be compatible with unix folder structure syntax
    (/ and space are replaced by _)
    """
    return string.replace('/', '_').replace(' ', '_')


def line_prepender(filename, line):
    """Prepend a certain string to txt (or csv file) and overwrite file

    Parameters
    ----------
    filename : :obj:`plt.str`
        Filename
    line : :obj:`plt.str`
        String to prepend

    """
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


def findclosest(array, value, checkoutside=False):
    """Find position in array of element closest to value

    Parameters
    ----------
    array : :obj:`np.ndarray`
        Array to be scanned
    value : :obj:`flot`
        Value to be found in array
    checkoutside : :obj:`bool`, optional
        Check if value is outside range of array (and return ``None``)

    Returns
    -------
    iclosest : :obj:`int`
        Index of closest element to value

    """
    iclosest = np.nanargmin(np.abs(array - value))
    if checkoutside and ((value < array[0]) or (value > array[-1])):
        iclosest = None
    return iclosest


def findclosest_2d(grid, value):
    r"""Find position in grid of element closest to value

    Parameters
    ----------
    grid : :obj:`list`
        List of two :obj:`np.ndarray` array representing thea
        axes of the grid
    value : :obj:`tuple` or :obj:`list`
        Values to be found in grid (as x and y locations)

    Returns
    -------
    iclosest : :obj:`list`
        Indeces of closest element to value

    """
    return [findclosest(grid[0], value[0]), findclosest(grid[1], value[1])]


def findclosest_scatter(scatters, value):
    """Find point in scatter of points closest to value

    Parameters
    ----------
    scatters : :obj:`np.ndarray`
        Scatter of points to be scanned (shaped as 2-d :obj:`np.ndarray` array)
    value : :obj:`tuple` or :obj:`list`
        Value to be found in scatter (as x and y locations)

    Returns
    -------
    iclosest : :obj:`list`
        Indeces of closest element to value

    """
    dist = np.sqrt((scatters[0] - value[0]) ** 2 +
                   (scatters[1] - value[1]) ** 2)
    iclosest = np.argmin(dist)

    return iclosest


def findindeces(array1, array2):
    """For each element of array2 return its position in array1
    (assumes that all elements in array2 are also present in array1)

    Parameters
    ----------
    array1 : :obj:`np.ndarray`
        Array to scan
    array2 : :obj:`tuple` or :obj:`list`
        Array with elements to whose position in array1 is returned

    Returns
    -------
    iclosest : :obj:`list`
        Indeces of closest element to value

    """
    return np.where(array2[:, None] == array1[None, :])[1]


def findvalid(array):
    """Finds index of first and last valid samples in array.

    The index of first and last samples are found based on the assumption that
    array has a number N of np.nan at start and M np.nan at the end and valid
    samples in between

    Parameters
    ----------
    array : :obj:`np.ndarray`
        Array to scan

    Returns
    -------
    istart : :obj:`int`
        Index of start
    iend : :obj:`int`
        Index of end

    """
    istart = np.where(~np.isnan(array))[0][0]
    iend = np.where(~np.isnan(array[istart:][::-1]))[0]
    if len(iend) == 0:
        iend = len(array) - 1
    else:
        iend = len(array) - iend[0] - 1
    return istart, iend

def unique_pairs(array):
    r"""Return pairs and indeces in array of size
    :math:`\lbrack n \times 2 \rbrack` that are unique


    Parameters
    ----------
    array : :obj:`np.ndarray`
        Array of size :math:`\lbrack n \times 2 \rbrack` to scan

    Returns
    -------
    unique : :obj:`np.ndarray`
        Array with only unique elements
    indices : :obj:`list`
        Array with only unique elements
    """
    # find uniques
    c = np.sort(array, axis=1)
    c_view = \
        np.ascontiguousarray(c).view(np.dtype((np.void,
                                               c.dtype.itemsize * c.shape[1])))
    _, indices = np.unique(c_view, return_index=True)

    # sort indices
    indices = indices[np.argsort(indices)]

    # bring unique elements back to original ordering
    return array[indices], np.sort(indices)
