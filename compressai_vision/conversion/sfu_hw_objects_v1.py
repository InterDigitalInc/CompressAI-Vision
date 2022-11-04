import glob
import os, re
from pathlib import Path
import fiftyone as fo
import csv

classmap = { # COCO-compatible
 0: 'person',
 1: 'bicycle',
 2: 'car',
 5: 'bus',
 7: 'truck',
 8: 'boat',
 13: 'bench',
 17: 'horse',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 32: 'sports ball',
 41: 'cup',
 56: 'chair',
 58: 'potted plant',
 60: 'dining table',
 63: 'laptop',
 67: 'cell phone',
 74: 'clock',
 77: 'teddy bear'
 }


def video_convert(basedir):
    """Converts video from YUV to lossless VP9

    Assumes this directory structure:

    ::

        basedir/
        ├── ClassA
        │   ├── Annotations
        │   │   ├── PeopleOnStreet [151 entries exceeds filelimit, not opening dir]
        │   │   └── Traffic [151 entries exceeds filelimit, not opening dir]
        │   ├── PeopleOnStreet_2560x1600_30_crop.yuv
        │   └── Traffic_2560x1600_30_crop.yuv
        ├── ClassB
        │   ├── Annotations
        │   │   ├── BasketballDrive [501 entries exceeds filelimit, not opening dir]
        │   │   ├── BQTerrace [601 entries exceeds filelimit, not opening dir]
        │   │   ├── Cactus [501 entries exceeds filelimit, not opening dir]
        │   │   ├── Kimono [241 entries exceeds filelimit, not opening dir]
        │   │   └── ParkScene [241 entries exceeds filelimit, not opening dir]
        │   ├── BasketballDrive_1920x1080_50Hz_8bit_P420.yuv
            etc. etc.
        ```

    Takes ``ClassA/Annotations/PeopleOnStreet_2560x1600_30_crop.yuv`` and
    converts it into ``ClassA/Annotations/PeopleOnStreet/video.webm`` at the lossless
    VP9 format.

    Same thing for all .yuv files found in the directory tree
    """
    r=re.compile('^(.*)\_(\d*)x(\d*)\_(\d*).*\.yuv')
    print("finding .yuv files from", basedir)
    for path in glob.glob(os.path.join(basedir,"*","*.yuv")):
        # print(path) # /home/sampsa/silo/interdigital/mock2/ClassA/BQTerrace_1920x1080_60Hz_8bit_P420.yuv
        fname=path.split(os.path.sep)[-1] # BQTerrace_1920x1080_60Hz_8bit_P420.yuv
        fname.split("_")[0] # BQTerrace
        # print(fname)
        m=r.match(fname)
        lis=m.groups()
        if len(lis) < 3:
            raise AssertionError("could not get x,y,fps from "+path)
        nametag, x, y, fps = lis # BQTerrace 1920 1080 60
        x=int(x)
        y=int(y)
        fps=int(fps)
        inpath=os.path.join(os.path.sep.join(path.split(os.path.sep)[0:-1]), nametag) # /home/sampsa/silo/interdigital/mock2/ClassA/BQTerrace
        # print(nametag, x, y, fps, inpath)
        st="ffmpeg -f rawvideo -pixel_format yuv420p -video_size {x}x{y} -i {input} -c:v libvpx-vp9 -lossless 1 -fps {fps} {output}".format(
            x=x, y=y, fps=fps, input=path, output=os.path.join(inpath, "Annotations", "video.webm")
        )
        print(st)
        os.system(st)
    print("video conversion done")

def sfu_txt_files_to_list(basedir):
    """Looks from basedir for files 

    ::

        something_NNN.txt

    where N is an integer.

    The frame numbering starts from "000".

    Returns a sorted list of tuples (index, filename), where indexes are taken (correctly) from the filenames.
    """
    p = Path(basedir)
    lis=[]
    r=re.compile(".*\_(\d\d\d)\.txt")
    for fname in glob.glob(str(p / "*.txt")):
        # print(fname)
        m=r.match(fname)
        try:
            itxt=m.group(1)
        except IndexError:
            print("inconsistent filename", fname)
            continue
        ind=int(itxt) # NOTE: index is taken from the filename
        lis.append((ind, fname))
    lis.sort()
    return lis


def read_detections(sample, lis):
    """reads detections into a video sample

    :param sample: fiftyone.Sample
    :param lis: a list of tuples with (frame_number, path)

    the file indicated by path has the following annotations format:

    class_num, x0, y0, w, h [all in relative coords]

    ::

        0 0.343100 0.912700 0.181200 0.167800
        0 0.696700 0.166200 0.120700 0.314900
        ...
    """
    for ind, fname in lis:
        # print("reading", fname)
        with open(fname,"r") as csvfile: # per frame
            dets=[]
            for line in csv.reader(csvfile, delimiter=' '): # per detection
                # print(line)
                n_class, x0, y0, w, h = line
                n_class = int(n_class)
                x0=float(x0) 
                y0=float(y0)
                w=float(w)
                h=float(h)
                # not top-left but bbox center coords
                x0=x0-w/2
                y0=y0-h/2
                label=classmap[n_class]
                bbox = [
                    x0, y0, w, h
                ]
                dets.append(
                    fo.Detection(
                        label=label, confidence=1., bounding_box=bbox
                    )
                )
            detections = fo.Detections(detections=dets)
            f=fo.Frame(
                detections=detections
            )
            sample.frames.add_frame(frame=f, frame_number=ind+1) # NOTE: frame numbering starts from 1
            # sample.frames.add_frame(frame=f, frame_number=ind+10) # TEST/DEBUG: inducing index mismatch


def register(dirname, name="sfu-hw-objects-v1"):
    """Register SFU-HW-Objects-v1 video directory into fiftyone

    ::

        ├── ClassA
        │   ├── Annotations
        │       ├── PeopleOnStreet / .txt files, video.webm
        │       └── Traffic / .txt files, video.webm
        ├── ClassB
        │   ├── Annotations
        │       ├── BasketballDrive 
        │       ├── BQTerrace 
        │       ├── Cactus 
        │       ├── Kimono
        │       └── ParkScene
        ...
        ...

    """
    classdirs=os.path.join(dirname,"Class*")
    print("searching for", classdirs)
    dirlist=glob.glob(classdirs)
    if len(dirlist) < 1:
        print("no directories found, will exit.  Check your path")
        return

    if name in fo.list_datasets():
        print("Dataset", name, "exists.  Will remove it first")
        fo.delete_dataset(name)
    dataset = fo.Dataset(name)
    print("Dataset", name, "created")

    for classdir in dirlist:
        # /path/to/ClassA
        print("In class directory", classdir)
        class_tag = classdir.split(os.path.sep)[-1] # ClassA
        annotation_dirs=os.path.join(classdir,"Annotations","*")
        print("searching for", annotation_dirs)
        annotation_dirlist=glob.glob(annotation_dirs)
        if len(annotation_dirlist) < 1:
            print("no directories found, will exit. Check your directory structure")
            return
        for annotations_dir in annotation_dirlist:
            # /path/to/ClassA/Annotations/PeopleOnStreet
            name_tag = annotations_dir.split(os.path.sep)[-1] # PeopleOnStreet
            filepath=os.path.join(annotations_dir, "video.webm")
            assert(os.path.exists(filepath)), "file "+filepath+" missing"
            print("registering video", filepath)
            custom_id = class_tag+"_"+name_tag
            video_sample=fo.Sample(
                media_type="video",
                filepath=filepath,
                class_tag=class_tag,
                name_tag=name_tag,
                custom_id=custom_id
            )
            # print(annotations_dir, class_tag, name_tag)
            n_filelist=sfu_txt_files_to_list(annotations_dir)
            #for n, filename in n_filelist:
            #    print(n, filename)            
            read_detections(video_sample, n_filelist)
            # video_sample now has the detections
            dataset.add_sample(video_sample)
            print("new video sample:", class_tag, name_tag, "with", len(n_filelist), "frames")

    dataset.persistent=True
    print("Dataset saved")


"""Example usage:

::

    import fiftyone as fo
    from fiftyone import ViewField as F
    import cv2
    
    from compressai_vision.conversion.sfu_hw_objects_v1 import video_convert
    from compressai_vision.conversion.sfu_hw_objects_v1 import register

    video_convert("/path/to/sfu_hw_objects_v1") # might take a while!
    # creates converted files into the directory structure:
    # ClassA/Annotations/PeopleOnStreets/video.webm
    # etc.
    # once that do you can do:
    register("/path/to/sfu_hw_objects_v1")
    # ..now it's all in fiftyone!

    dataset=fo.load_dataset("sfu_hw_objects_v1")
    # each sample in the dataset corresponds to video:
    # i.e. SAMPLE = VIDEO
    sample=dataset.first()
    print(sample)
    # each sample has Frame objects -> detections for each frame
    # i.e. FRAME = BBOXES, etc.
    print(sample.frames[1])
    # accessing different samples, aka videos
    print(dataset.distinct("filepath"))
    tmpset=ds[(F("class_tag") == "ClassA") & F("name_tag") == "Traffic"]
    print(tmpset)

    # accessing frame images
    vid=cv2.VideoCapture(sample.filepath)
    # go to frame number 11
    ok = vid.set(cv2.CAP_PROP_POS_FRAMES, 11-1)
    ok, arr = vid.read() # BGR image in arr
    # accessing corresponding detection data:
    sample.frames[11].detections

"""
