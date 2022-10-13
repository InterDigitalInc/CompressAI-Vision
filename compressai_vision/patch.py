"""monkey-patching for https://github.com/voxel51/fiftyone/issues/2096
"""

# import importhook # this module simply ...ks up everything (at least torch imports)
from datetime import datetime

import fiftyone
import fiftyone.core.odm as foo


def _make_sample_collection_name(patches=False, frames=False, clips=False):
    # as per
    # https://github.com/voxel51/fiftyone/blob/develop/fiftyone/core/dataset.py#L5616
    # https://github.com/voxel51/fiftyone/issues/2096
    #
    conn = foo.get_db_conn()
    now = datetime.now()

    if patches:
        prefix = "patches"
    elif frames:
        prefix = "frames"
    elif clips:
        prefix = "clips"
    else:
        prefix = "samples"

    create_name = lambda timestamp: ".".join([prefix, timestamp])
    name = create_name(now.strftime("%Y.%m.%d.%H.%M.%S.%f"))
    # print(">PATCH") # checking that this method is picked up..
    return name


# apply patches
# print(">PATCHING!")
fiftyone.core.dataset._make_sample_collection_name = _make_sample_collection_name
