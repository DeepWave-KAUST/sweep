import numpy as np

def split_model(m, sources, receivers, one_side_expand=50):
    """Split the model based on the sources and receivers locations for saving memory and computational resources.

    Args:
        m (2d array): The model to be split, shape (nz, nx).
        sources (2d array): The source locations, shape (nshots, 2).
        receivers (3d array): The receiver locations, shape (nshots, nreceivers, 2).
        one_side_expand (int): The number of grids to expand for one side.

    Returns:
        tuple: A tuple containing:
            - m_split (array): The split velocity model, shape (nshots, nz, nx).
            - left (array): The leftmost index of the split model for each shot.
            - right (array): The rightmost index of the split model for each shot.
    """
    ori_domain = m[0].shape
    assert sources.shape[0] == receivers.shape[0], "Sources and receivers must have the same number of shots"

    m_split = []
    left = []
    right = []

    sources_moved = sources.copy()
    receivers_moved = receivers.copy()

    # Calculate the max length between the sources and receivers for each shot
    # and determine the leftmost and rightmost points
    # max_length = np.abs(np.max(receivers[..., 0], axis=1) - sources[..., 0]).max()
    max_length = np.max(np.abs(receivers[..., 0] - sources[..., 0][..., None]), axis=1).max()
    for shot in range(sources.shape[0]):
        srcx = sources[shot, ..., 0]
        recx = receivers[shot, ..., 0]

        leftmost_recx = np.min(recx)
        rightmost_recx = np.max(recx)

        leftmost = np.min(np.array([leftmost_recx, srcx]), axis=0)
        rightmost = np.max(np.array([rightmost_recx, srcx]), axis=0)

        # Expand the leftmost and rightmost points by the specified number of grids
        expand_left = one_side_expand

        if leftmost - expand_left < 0:
            left_start = 0
            expand_left = leftmost
            expand_right = one_side_expand*2 - expand_left
        else:
            left_start = leftmost - expand_left
            expand_right = one_side_expand

        right_end = left_start + expand_left + max_length + expand_right

        if right_end > ori_domain[1]: # if the right end exceeds the domain size, try to adjust

            right_end = ori_domain[1]
            expand_right = right_end - rightmost
            expand_left = one_side_expand * 2 - expand_right
            left_start = leftmost - expand_left
            assert left_start >= 0, f'Left start {left_start} is out of bounds for the domain size {ori_domain[1]}'
            assert right_end <= ori_domain[1], f'Right end {right_end} is out of bounds for the domain size {ori_domain[1]}'

        m_split.append([_m[:, left_start:right_end] for _m in m])
        left.append(left_start)
        right.append(right_end)

        sources_moved = sources_moved.at[shot, ..., 0].subtract(left_start)
        receivers_moved = receivers_moved.at[shot, ..., 0].subtract(left_start)
    try:
        m_split = np.stack(m_split, axis=0)
    except ValueError as e:
        print(f"Error stacking m_split: {e}")
    left = np.array(left)
    right = np.array(right)

    return m_split, left, right, sources_moved, receivers_moved