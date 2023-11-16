# cython_bbox

cython_bbox is widely used in object detection tasks. To my best knowledge, it was first implemented in [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn). Since then, almost all object detection projects use the source code directly.

In order to use it in standalone code snippets or small projects, I make it a pypi module. The `cython_bbox.pyx` is totally borrowed from [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn). Thanks [RBG](http://www.rossgirshick.info/)!

## install

```
pip install cython_bbox
```

## usage


```
from cython_bbox import bbox_overlaps
import numpy as np

gt = np.random.random((5, 4))
dt = np.random.random((10, 4))

overlaps = bbox_overlaps(
        np.ascontiguousarray(dt, dtype=np.float),
        np.ascontiguousarray(gt, dtype=np.float)
    )

```
