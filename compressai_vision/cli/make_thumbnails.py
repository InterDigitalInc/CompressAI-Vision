# Copyright (c) 2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Use this stub for adding new cli commands
"""
import os


def add_subparser(subparsers, parents):
    subparser = subparsers.add_parser(
        "make-thumbnails", parents=parents, help="Create 'side-data' videos that work with the fiftyone webapp"
    )
    some_group = subparser.add_argument_group("required arguments")
    some_group.add_argument(
        "--dataset-name",
        action="store",
        type=str,
        required=True,
        default=None,
        help="name of the dataset",
    )
    some_group.add_argument(
        "--force",
        action="store",
        type=str,
        required=False,
        default=False,
        help="encode files even if they already existed",
    )


    
def main(p):
    """https://voxel51.com/docs/fiftyone/user_guide/app.html#multiple-media-fields

    https://voxel51.com/docs/fiftyone/api/fiftyone.utils.video.html#fiftyone.utils.video.reencode_videos
    """
    # fiftyone
    #if not p.y:
    #    input("press enter to continue.. ")
    #    print()
    # p.some_dir = os.path.expanduser(p.some_dir) # correct path in the case user uses POSIX "~"
    print("importing fiftyone")
    import fiftyone as fo
    import fiftyone.utils.video as fouv
    print("fiftyone imported")
    print()
    try:
        dataset=fo.load_dataset(p.dataset_name)
    except ValueError:
        print("Sorry, could not find dataset", p.dataset_name)
    assert dataset.media_type == "video", "this command works only for video datasets"
    print("Will encode webapp-compatible versions of the videos")
    print("This WILL take a while!")
    if not p.y:
        input("press enter to continue.. ")
    for sample in dataset.iter_samples(progress=True):
        sample_dir = os.path.dirname(sample.filepath)
        output_path = os.path.join(sample_dir, "web_"+sample.filename)
        if (not p.force) and os.path.isfile(output_path):
            print("WARNING: file", output_path, "already exists - will skip")
            continue
        print("\nRe-encoding", sample.filepath,"to", output_path)
        fouv.reencode_video(sample.filepath, output_path)
        sample["web_filepath"] = output_path
        sample.save()
