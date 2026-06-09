# Packages
import cv2
import numpy as np

# Funcktions
from skimage.morphology import skeletonize

# Custom package
import cv_algorithms


# ==============================================================================
# This file contains a series of functions to process an array before extrac-
# ting faults. This includes functions for:
# (1) 阈值化
# (2) 骨架化
# (3) 为连接的组件贴标签
# (4) 删除组件
# (5) 转换为点。
# ==============================================================================


# ******************************************************************************
# (1) THRESHOLDING
# 允许您以不同方式对数据进行阈值设置的几个函数
# ******************************************************************************

def simple_threshold_binary(arr, threshold):
    """ 将阈值数组转换为二进制数组

    Parameters
    ----------
    arr : np.array
        我们用阈值二值化的输入数组

    threshold : int, float
        用于二值化输入数组的阈值

    Returns
    -------
    arr
        Binarized output array (type: uint8)
    """

    # Assertions
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"
    assert isinstance(threshold, int) or isinstance(threshold, float), "Threshold is neither int nor float"

    # 计算
    arr = np.where(arr > threshold, 1, 0)
    arr = np.uint8(arr)

    return arr


def adaptive_threshold(arr):
    """ 使用自适应阈值（二进制+Otsu）将阈值数组数组转换为二进制数组

    Parameters
    ----------

    arr : np.array
        我们用阈值二值化的输入数组

    Returns
    -------
    arr
        二值化输出数组（类型：uint8）
    """

    # Assertion
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"

    # Calculation
    # Scale to [0,1]
    arr = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
    # Scale to [0,255]
    arr = 255 * arr
    # Create image
    image = cv2.resize(arr.astype('uint8'), dsize=(arr.shape[1], arr.shape[0]))
    # 应用自适应阈值
    _, arr = cv2.threshold(image,
                           0,
                           1,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 转换回数字派数组
    arr = np.uint8(arr)

    return arr


# ******************************************************************************
# (2) SKELETONIZE
# 几个允许您对数据进行骨架化的函数，即减少到一个像素粗的线条
# ******************************************************************************

def skeleton_scipy(arr):
    """ 来自 SciPy 的基本骨架化功能

    Parameters
    ----------

    arr : np.array
        Input array

    Returns
    -------
    arr
        Output array
    """

    # Assertion
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"

    return skeletonize(arr)


def skeleton_guo_hall(arr):
    """ 优化了cv_algorithms的骨架化功能 （https://github.com/ulikoehler/cv_algorithms）

    Parameters
    ----------

    arr : np.array
        Input array

    Returns
    -------
    arr
        Output array
    """

    # Assertion
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"

    # Calculation
    arr = cv_algorithms.guo_hall(arr)

    # 正确的边缘效果
    arr[0, :] = arr[1, :]
    arr[-1, :] = arr[-2, :]
    arr[:, 0] = arr[:, 1]
    arr[:, -1] = arr[:, -2]

    return arr


# ******************************************************************************
# (3) CONNECTED COMPONENTS
# 在阵列中标记连接组件的功能
# ******************************************************************************

def connected_components(arr):
    """ 标记连接的组件

    Parameters
    ----------

    arr : np.array
        Input array

    Returns
    -------
    ret
        Output array
    markers
        Components
    """

    # Assertion
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"

    # Calculation
    ret, markers = cv2.connectedComponents(arr)

    return ret, markers


# ******************************************************************************
# (4) REMOVAL
# 删除某些组件的几个功能
# ******************************************************************************

def remove_small_regions(arr, size):
    """移除低于特定尺寸的组件

    Parameters
    ----------

    arr : np.array
        Input array
    size : int

    Returns
    -------
    arr
        Output array
    """
    # Assertion
    assert isinstance(arr, np.ndarray), "Input array is not a NumPy array"
    assert isinstance(arr, int), "Input size is not an integer "

    # 查找所有连接的组件（图像中的白色斑点）
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(arr, connectivity=8)

    # connectedComponentswithStats 生成每个单独的组件，
    # 其中包含每个组件的信息，例如大小.
    # 以下部分只是取出背景，这也被认为是一个组件，但大多数时候我们不希望这样。
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # 我们要保留的粒子的最小尺寸（像素数）在这里，
    # 它是一个固定值，但您可以根据需要设置它，例如大小的平均值或其他任何

    # your answer image
    arr = np.zeros((output.shape))
    # 对于映像中的每个组件，仅当它高于min_size时才保留它
    for i in range(0, nb_components):
        if sizes[i] >= size:
            arr[output == i + 1] = 255

    # Convert to uint8
    arr = np.uint8(arr)

    return arr


def remove_large_regions(arr, size):
    """ 移除超过特定尺寸的组件

    Parameters
    ----------

    arr : np.array
        Input array
    size : int

    Returns
    -------
    arr
        Output array
    """
    # Assertion
    assert isinstance(arr, np.ndarray), "Input array is not a NumPy array"
    assert isinstance(arr, int), "Input size is not an integer "

    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(arr, connectivity=8)

    # connectedComponentswithStats yields every seperated component with
    # information on each of them, such as size
    # the following part is just taking out the background which is also
    # considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of
    # the sizes or whatever

    # your answer image
    arr = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] <= size:
            arr[output == i + 1] = 255

    # Convert to uint8
    arr = np.uint8(arr)

    return arr


# ******************************************************************************
# (5) CONVERSION
# 将数组转换为点 （x，y） 的函数
# ******************************************************************************

def array_to_points(arr):
    """ 将数组转换为点 （x，y） 的函数

    Parameters
    ----------
    arr : np.array
        Input array that we binarize with threshold


    Returns
    -------
    arr
        Output array (points)
    """

    # Assertions
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"

    # Calculation
    n = np.count_nonzero(arr)
    points = np.zeros((n, 2))
    (points[:, 1], points[:, 0]) = np.where(arr != 0)

    return points



