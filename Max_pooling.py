import numpy as np


def get_pools(img, pool_size, stride):
    pools = []

    for i in range(0, img.shape[0], stride):
        for j in range(0, img.shape[1], stride):
            mat = img[i: i + pool_size, j: j + pool_size]
            if mat.shape == (pool_size, pool_size)
                pools.append(mat)

    return pools


def max_pooling(pools):
    num_pools = pools.shape[0]
    tgt_shape = (int(np.sqrt(num_pools)), int(np.sqrt(num_pools)))
    pooled = []

    for pool in pools:
        pooled.append(np.max(pool))

    return np.array(pooled).reshape(tgt_shape)



