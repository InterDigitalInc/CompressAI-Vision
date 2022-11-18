import configparser
import csv
import glob
import os

# from pathlib import Path
import fiftyone as fo


def read_detections(sample, fname):
    """
    :param sample: fiftyone.Sample
    :param fname: frame-by-frame annotations

    TVD format

    ::

        [Frame_Index, Object_ID, Top_left_x, Top_left_y, Width, Height, Confidence, 3D_x, 3D_y] ?

    or is one of them a class label..?  what label set?
    note that these are abs coordinates


    Example:

    ::


        1,1,193,686,125,331,1,1,1
        2,1,193,686,124,330,1,1,1
        3,1,194,686,124,330,1,1,1
        4,1,197,684,116,339,1,1,1
        5,1,194,684,121,330,1,1,1
        6,1,199,685,113,335,1,1,1
        ...
        543,1,645,855,47,125,1,1,1
        544,1,646,860,48,118,1,1,1
        1,3,746,894,1098,106,0,9,1
        2,3,746,894,1098,106,0,9,1
        ...

    i.e. note that frame indexes can start again from 1

    """
    with open(fname, "r") as csvfile:  # per frame
        dets_per_frame = {}
        for line in csv.reader(csvfile, delimiter=","):  # per detection
            (
                Frame_Index,
                Object_ID,
                Top_left_x,
                Top_left_y,
                Width,
                Height,
                Confidence,
                _,
                _,
            ) = line
            Frame_Index = int(Frame_Index)
            object_id = int(Object_ID)
            x0 = float(Top_left_x) / sample.metadata.frame_width
            y0 = float(Top_left_y) / sample.metadata.frame_height
            w = float(Width) / sample.metadata.frame_width
            h = float(Height) / sample.metadata.frame_height
            confidence = float(Confidence)
            # bbox center coords:
            # x0=x0-w/2
            # y0=y0-h/2
            # label=classmap[n_class]
            bbox = [x0, y0, w, h]
            if Frame_Index not in dets_per_frame:
                dets_per_frame[Frame_Index] = []
            dets_per_frame[Frame_Index].append(
                fo.Detection(
                    # label=label,  # no label? # for MOT we can always substitute the label with the id..!
                    label="#" + str(object_id),
                    # index=object_id,
                    confidence=confidence,
                    bounding_box=bbox,
                )
            )

    for Frame_Index, dets in dets_per_frame.items():
        detections = fo.Detections(detections=dets)
        f = fo.Frame(detections=detections)
        sample.frames.add_frame(frame=f, frame_number=Frame_Index)


def register(dirname, name="tvd-object-tracking-v1"):
    """Register tencent video dataset (TVD), object tracking subset.

    The directory structure for this looks like:

    ::

        dirname/
            |
            ├── TVD-01
            │   ├── gt
            │   │   └── gt.txt
            │   ├── img1
            │   └── seqinfo.ini
            ├── TVD-01.mp4
            ├── TVD-02
            │   ├── gt
            │   │   └── gt.txt
            │   ├── img1
            │   └── seqinfo.ini
            ├── TVD-02.mp4
            ├── TVD-03
            │   ├── gt
            │   │   └── gt.txt
            │   ├── img1
            │   └── seqinfo.ini
            └── TVD-03.mp4

    """
    dirs = os.path.join(dirname, os.path.join("TVD-*", ""))  # in order to get dirs only
    print("searching for", dirs)
    dirlist = glob.glob(dirs)
    if len(dirlist) < 1:
        print("no directories found, will exit.  Check your path")
        return -1
    # print(dirlist)
    if name in fo.list_datasets():
        print("Dataset", name, "exists.  Will remove it first")
        fo.delete_dataset(name)
    dataset = fo.Dataset(name)
    print("Dataset", name, "created")
    for dir_ in dirlist:
        # /path/to/TVD-01/
        print("\nIn directory", dir_)
        tag = dir_.split(os.path.sep)[-2]
        print(tag)  # TVD-01

        gt_file = os.path.join(dir_, "gt", "gt.txt")
        print(gt_file)
        assert os.path.isfile(gt_file), "can't find " + gt_file

        video_file = os.path.join(dirname, tag + ".mp4")
        print(video_file)
        assert os.path.isfile(video_file), "can't find " + video_file

        ini_file = os.path.join(dirname, tag, "seqinfo.ini")
        print(ini_file)
        assert os.path.isfile(ini_file), "can't find " + ini_file

        config = configparser.ConfigParser()
        config.read(ini_file)
        seq = config["Sequence"]
        # frameRate = int(seq["frameRate"])
        imWidth = int(seq["imWidth"])
        imHeight = int(seq["imHeight"])

        metadata = fo.VideoMetadata(frame_width=imWidth, frame_height=imHeight)
        video_sample = fo.Sample(
            media_type="video", filepath=video_file, metadata=metadata, name_tag=tag
        )

        print("importing detections for", video_file, "from", gt_file)
        read_detections(video_sample, gt_file)
        # video_sample now has the detections
        dataset.add_sample(video_sample)
    dataset.persistent = True
    print("\nDONE!")
