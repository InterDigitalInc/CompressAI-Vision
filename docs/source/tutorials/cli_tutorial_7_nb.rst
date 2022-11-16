In this tutorial you will learn how to:

-  Download and register video datasets
-  Convert and import the ``sfu-hw-objects-v1`` raw custom video data
   format
-  Play around with video datasets, visualize frames and detection
   results
-  Evaluate a video dataset

In chapter 2 of this tutorial you learned how to download and register
datasets to fiftyone with the ``compressai-vision register`` command.

Exactly the same command works for video datasets:

.. code:: bash

    compressai-vision download --dataset-name=quickstart-video --y


.. code-block:: text

    importing fiftyone
    fiftyone imported
    
    WARNING: downloading ALL images.  You might want to use the --lists option to download only certain images
    Using list files:     None
    Number of images:     ?
    Database name   :     quickstart-video
    Subname/split   :     None
    Target dir      :     None
    
    Dataset already downloaded
    Loading existing dataset 'quickstart-video'. To reload from disk, either delete the existing dataset or provide a custom `dataset_name` to use


If you have your video dataset arranged in one of the standard `video
data formats supported by
fiftyone <https://voxel51.com/docs/fiftyone/api/fiftyone.types.dataset_types.html>`__,
you’re good to go.

Manipulating and visualizing video datasets from python works a bit
different to image datasets. For this, please see the end of this
tutorial.

Next we will import a raw custom dataset, namely the
`sfu-hw-objects-v1 <http://dx.doi.org/10.17632/hwm673bv4m.1>`__ into
fiftyone.

This format consists raw YUV video files and annotations. Let’s see how
the folder structure is roughly organized. We’ll be using in this
tutorial a “mock” version of the dataset with only two video classes:

.. code:: bash

    tree {path_to_sfu_hw_objects_v1} --filelimit=10 | cat


.. code-block:: text

    /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1
    ├── ClassC
    │   ├── Annotations
    │   │   └── BasketballDrill [502 entries exceeds filelimit, not opening dir]
    │   └── BasketballDrill_832x480_50Hz_8bit_P420.yuv
    └── ClassX
        ├── Annotations
        │   └── BasketballDrill
        │       ├── BasketballDrill_832x480_50_seq_001.txt
        │       ├── BasketballDrill_832x480_50_seq_002.txt
        │       ├── BasketballDrill_832x480_50_seq_003.txt
        │       ├── BasketballDrill_832x480_50_seq_004.txt
        │       └── BasketballDrill_832x480_object.list
        └── BasketballDrill_832x480_50Hz_8bit_P420.yuv -> /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassC/BasketballDrill_832x480_50Hz_8bit_P420.yuv
    
    6 directories, 7 files


Importing mpeg-vcom custom datasets (for more info see Dataset section
of the documentation) can be done with ``import-custom`` command. For
``sfu-hw-objects-v1`` it also converts on-the-fly the raw YUV images
into proper video format:

.. code:: bash

    compressai-vision import-custom --dataset-type=sfu-hw-objects-v1 --dir={path_to_sfu_hw_objects_v1} --y


.. code-block:: text

    importing fiftyone
    fiftyone imported
    WARNING: dataset sfu-hw-objects-v1 already exists: will delete and rewrite
    
    Importing a custom video format into fiftyone
    
    Dataset type           :  sfu-hw-objects-v1
    Dataset root directory :  /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1
    
    finding .yuv files from /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1
    ffmpeg -y -f rawvideo -pixel_format yuv420p -video_size 832x480 -i /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassC/BasketballDrill_832x480_50Hz_8bit_P420.yuv -an -c:v h264 -q 0 /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassC/Annotations/BasketballDrill/video.mp4
    ffmpeg version 4.2.7-0ubuntu0.1 Copyright (c) 2000-2022 the FFmpeg developers
      built with gcc 9 (Ubuntu 9.4.0-1ubuntu1~20.04.1)
      configuration: --prefix=/usr --extra-version=0ubuntu0.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-avresample --disable-filter=resample --enable-avisynth --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librsvg --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwavpack --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-nvenc --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared
      libavutil      56. 31.100 / 56. 31.100
      libavcodec     58. 54.100 / 58. 54.100
      libavformat    58. 29.100 / 58. 29.100
      libavdevice    58.  8.100 / 58.  8.100
      libavfilter     7. 57.100 /  7. 57.100
      libavresample   4.  0.  0 /  4.  0.  0
      libswscale      5.  5.100 /  5.  5.100
      libswresample   3.  5.100 /  3.  5.100
      libpostproc    55.  5.100 / 55.  5.100
    [0;35m[rawvideo @ 0x561a0d3c17c0] [0m[0;33mEstimating duration from bitrate, this may be inaccurate
    [0mInput #0, rawvideo, from '/home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassC/BasketballDrill_832x480_50Hz_8bit_P420.yuv':
      Duration: 00:00:20.04, start: 0.000000, bitrate: 119808 kb/s
        Stream #0:0: Video: rawvideo (I420 / 0x30323449), yuv420p, 832x480, 119808 kb/s, 25 tbr, 25 tbn, 25 tbc
    Stream mapping:
      Stream #0:0 -> #0:0 (rawvideo (native) -> h264 (libx264))
    Press [q] to stop, [?] for help
    [1;36m[libx264 @ 0x561a0d3cf300] [0musing cpu capabilities: MMX2 SSE2Fast SSSE3 SSE4.2 AVX FMA3 BMI2 AVX2
    [1;36m[libx264 @ 0x561a0d3cf300] [0mprofile High, level 3.0
    [1;36m[libx264 @ 0x561a0d3cf300] [0m264 - core 155 r2917 0a84d98 - H.264/MPEG-4 AVC codec - Copyleft 2003-2018 - http://www.videolan.org/x264.html - options: cabac=1 ref=3 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 threads=12 lookahead_threads=2 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=2 keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=crf mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
    Output #0, mp4, to '/home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassC/Annotations/BasketballDrill/video.mp4':
      Metadata:
        encoder         : Lavf58.29.100
        Stream #0:0: Video: h264 (libx264) (avc1 / 0x31637661), yuv420p, 832x480, q=-1--1, 25 fps, 12800 tbn, 25 tbc
        Metadata:
          encoder         : Lavc58.54.100 libx264
        Side data:
          cpb: bitrate max/min/avg: 0/0/0 buffer size: 0 vbv_delay: -1
    frame=  501 fps=143 q=-1.0 Lsize=    3979kB time=00:00:19.92 bitrate=1636.2kbits/s speed=5.67x    
    video:3972kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.169325%
    [1;36m[libx264 @ 0x561a0d3cf300] [0mframe I:3     Avg QP:22.61  size: 56539
    [1;36m[libx264 @ 0x561a0d3cf300] [0mframe P:126   Avg QP:24.67  size: 17479
    [1;36m[libx264 @ 0x561a0d3cf300] [0mframe B:372   Avg QP:28.66  size:  4556
    [1;36m[libx264 @ 0x561a0d3cf300] [0mconsecutive B-frames:  1.0%  0.0%  0.0% 99.0%
    [1;36m[libx264 @ 0x561a0d3cf300] [0mmb I  I16..4: 13.3% 37.2% 49.4%
    [1;36m[libx264 @ 0x561a0d3cf300] [0mmb P  I16..4:  0.1% 11.2%  6.3%  P16..4: 42.9% 16.1% 11.6%  0.0%  0.0%    skip:11.7%
    [1;36m[libx264 @ 0x561a0d3cf300] [0mmb B  I16..4:  0.0%  0.7%  0.4%  B16..8: 35.6%  9.2%  3.6%  direct: 3.0%  skip:47.6%  L0:43.7% L1:43.7% BI:12.7%
    [1;36m[libx264 @ 0x561a0d3cf300] [0m8x8 transform intra:60.9% inter:67.7%
    [1;36m[libx264 @ 0x561a0d3cf300] [0mcoded y,uvDC,uvAC intra: 87.9% 88.2% 66.7% inter: 22.5% 18.1% 4.7%
    [1;36m[libx264 @ 0x561a0d3cf300] [0mi16 v,h,dc,p: 57% 13%  8% 22%
    [1;36m[libx264 @ 0x561a0d3cf300] [0mi8 v,h,dc,ddl,ddr,vr,hd,vl,hu: 11%  8%  9%  7% 19% 17% 10%  9%  9%
    [1;36m[libx264 @ 0x561a0d3cf300] [0mi4 v,h,dc,ddl,ddr,vr,hd,vl,hu: 14%  9% 13%  7% 19% 15%  8%  7%  6%
    [1;36m[libx264 @ 0x561a0d3cf300] [0mi8c dc,h,v,p: 48% 17% 22% 13%
    [1;36m[libx264 @ 0x561a0d3cf300] [0mWeighted P-Frames: Y:0.0% UV:0.0%
    [1;36m[libx264 @ 0x561a0d3cf300] [0mref P L0: 44.5% 27.1% 14.9% 13.6%
    [1;36m[libx264 @ 0x561a0d3cf300] [0mref B L0: 85.6% 10.0%  4.4%
    [1;36m[libx264 @ 0x561a0d3cf300] [0mref B L1: 94.4%  5.6%
    [1;36m[libx264 @ 0x561a0d3cf300] [0mkb/s:1623.41
    ffmpeg -y -f rawvideo -pixel_format yuv420p -video_size 832x480 -i /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassX/BasketballDrill_832x480_50Hz_8bit_P420.yuv -an -c:v h264 -q 0 /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassX/Annotations/BasketballDrill/video.mp4
    ffmpeg version 4.2.7-0ubuntu0.1 Copyright (c) 2000-2022 the FFmpeg developers
      built with gcc 9 (Ubuntu 9.4.0-1ubuntu1~20.04.1)
      configuration: --prefix=/usr --extra-version=0ubuntu0.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-avresample --disable-filter=resample --enable-avisynth --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librsvg --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwavpack --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-nvenc --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared
      libavutil      56. 31.100 / 56. 31.100
      libavcodec     58. 54.100 / 58. 54.100
      libavformat    58. 29.100 / 58. 29.100
      libavdevice    58.  8.100 / 58.  8.100
      libavfilter     7. 57.100 /  7. 57.100
      libavresample   4.  0.  0 /  4.  0.  0
      libswscale      5.  5.100 /  5.  5.100
      libswresample   3.  5.100 /  3.  5.100
      libpostproc    55.  5.100 / 55.  5.100
    [0;35m[rawvideo @ 0x559c0f4467c0] [0m[0;33mEstimating duration from bitrate, this may be inaccurate
    [0mInput #0, rawvideo, from '/home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassX/BasketballDrill_832x480_50Hz_8bit_P420.yuv':
      Duration: 00:00:20.04, start: 0.000000, bitrate: 119808 kb/s
        Stream #0:0: Video: rawvideo (I420 / 0x30323449), yuv420p, 832x480, 119808 kb/s, 25 tbr, 25 tbn, 25 tbc
    Stream mapping:
      Stream #0:0 -> #0:0 (rawvideo (native) -> h264 (libx264))
    Press [q] to stop, [?] for help
    [1;36m[libx264 @ 0x559c0f454300] [0musing cpu capabilities: MMX2 SSE2Fast SSSE3 SSE4.2 AVX FMA3 BMI2 AVX2
    [1;36m[libx264 @ 0x559c0f454300] [0mprofile High, level 3.0
    [1;36m[libx264 @ 0x559c0f454300] [0m264 - core 155 r2917 0a84d98 - H.264/MPEG-4 AVC codec - Copyleft 2003-2018 - http://www.videolan.org/x264.html - options: cabac=1 ref=3 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 threads=12 lookahead_threads=2 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=2 keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=crf mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
    Output #0, mp4, to '/home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassX/Annotations/BasketballDrill/video.mp4':
      Metadata:
        encoder         : Lavf58.29.100
        Stream #0:0: Video: h264 (libx264) (avc1 / 0x31637661), yuv420p, 832x480, q=-1--1, 25 fps, 12800 tbn, 25 tbc
        Metadata:
          encoder         : Lavc58.54.100 libx264
        Side data:
          cpb: bitrate max/min/avg: 0/0/0 buffer size: 0 vbv_delay: -1
    frame=  501 fps=131 q=-1.0 Lsize=    3979kB time=00:00:19.92 bitrate=1636.2kbits/s speed= 5.2x    
    video:3972kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.169325%
    [1;36m[libx264 @ 0x559c0f454300] [0mframe I:3     Avg QP:22.61  size: 56539
    [1;36m[libx264 @ 0x559c0f454300] [0mframe P:126   Avg QP:24.67  size: 17479
    [1;36m[libx264 @ 0x559c0f454300] [0mframe B:372   Avg QP:28.66  size:  4556
    [1;36m[libx264 @ 0x559c0f454300] [0mconsecutive B-frames:  1.0%  0.0%  0.0% 99.0%
    [1;36m[libx264 @ 0x559c0f454300] [0mmb I  I16..4: 13.3% 37.2% 49.4%
    [1;36m[libx264 @ 0x559c0f454300] [0mmb P  I16..4:  0.1% 11.2%  6.3%  P16..4: 42.9% 16.1% 11.6%  0.0%  0.0%    skip:11.7%
    [1;36m[libx264 @ 0x559c0f454300] [0mmb B  I16..4:  0.0%  0.7%  0.4%  B16..8: 35.6%  9.2%  3.6%  direct: 3.0%  skip:47.6%  L0:43.7% L1:43.7% BI:12.7%
    [1;36m[libx264 @ 0x559c0f454300] [0m8x8 transform intra:60.9% inter:67.7%
    [1;36m[libx264 @ 0x559c0f454300] [0mcoded y,uvDC,uvAC intra: 87.9% 88.2% 66.7% inter: 22.5% 18.1% 4.7%
    [1;36m[libx264 @ 0x559c0f454300] [0mi16 v,h,dc,p: 57% 13%  8% 22%
    [1;36m[libx264 @ 0x559c0f454300] [0mi8 v,h,dc,ddl,ddr,vr,hd,vl,hu: 11%  8%  9%  7% 19% 17% 10%  9%  9%
    [1;36m[libx264 @ 0x559c0f454300] [0mi4 v,h,dc,ddl,ddr,vr,hd,vl,hu: 14%  9% 13%  7% 19% 15%  8%  7%  6%
    [1;36m[libx264 @ 0x559c0f454300] [0mi8c dc,h,v,p: 48% 17% 22% 13%
    [1;36m[libx264 @ 0x559c0f454300] [0mWeighted P-Frames: Y:0.0% UV:0.0%
    [1;36m[libx264 @ 0x559c0f454300] [0mref P L0: 44.5% 27.1% 14.9% 13.6%
    [1;36m[libx264 @ 0x559c0f454300] [0mref B L0: 85.6% 10.0%  4.4%
    [1;36m[libx264 @ 0x559c0f454300] [0mref B L1: 94.4%  5.6%
    [1;36m[libx264 @ 0x559c0f454300] [0mkb/s:1623.41
    video conversion done
    searching for /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/Class*
    Dataset sfu-hw-objects-v1 exists.  Will remove it first
    Dataset sfu-hw-objects-v1 created
    
    In class directory /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassC
    searching for /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassC/Annotations/*
    --> registering video /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassC/Annotations/BasketballDrill/video.mp4
    --> registered new video sample: ClassC BasketballDrill with 500 frames
    
    In class directory /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassX
    searching for /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassX/Annotations/*
    --> registering video /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassX/Annotations/BasketballDrill/video.mp4
    --> registered new video sample: ClassX BasketballDrill with 4 frames
    
    Dataset saved


In order to demonstrate how video datasets are used, let’s continue in
python notebook:

.. code:: ipython3

    import cv2
    import matplotlib.pyplot as plt
    import fiftyone as fo
    from fiftyone import ViewField as F
    from math import floor

.. code:: ipython3

    dataset=fo.load_dataset("sfu-hw-objects-v1")

.. code:: ipython3

    dataset




.. parsed-literal::

    Name:        sfu-hw-objects-v1
    Media type:  video
    Num samples: 2
    Persistent:  True
    Tags:        []
    Sample fields:
        id:         fiftyone.core.fields.ObjectIdField
        filepath:   fiftyone.core.fields.StringField
        tags:       fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
        metadata:   fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.VideoMetadata)
        media_type: fiftyone.core.fields.StringField
        class_tag:  fiftyone.core.fields.StringField
        name_tag:   fiftyone.core.fields.StringField
        custom_id:  fiftyone.core.fields.StringField
    Frame fields:
        id:           fiftyone.core.fields.ObjectIdField
        frame_number: fiftyone.core.fields.FrameNumberField
        detections:   fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)



In contrast to image datasets where each sample was an image, now a
sample corresponds to a video:

.. code:: ipython3

    dataset.first()




.. parsed-literal::

    <Sample: {
        'id': '636cfe83dd3cdcdcd97f0f64',
        'media_type': 'video',
        'filepath': '/home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassC/Annotations/BasketballDrill/video.mp4',
        'tags': BaseList([]),
        'metadata': None,
        'class_tag': 'ClassC',
        'name_tag': 'BasketballDrill',
        'custom_id': 'ClassC_BasketballDrill',
        'frames': <Frames: 500>,
    }>



There is a reference to the video file and a ``Frames`` object,
encapsulating ground truths etc. data for each and every frame. For
``sfu-hw-objects-v1`` in particular, ``class_tag`` corresponds to the
class directories (ClassA, ClassB, etc.), while ``name_tag`` to the
video descriptive names (BasketballDrill, Traffic, PeopleOnStreeet,
etc.). Let’s pick a certain video sample:

.. code:: ipython3

    sample = dataset[ (F("name_tag") == "BasketballDrill") & (F("class_tag") == "ClassC") ].first()

Take a look at the first frame ground truth detections (note that frame
indices start from 1):

.. code:: ipython3

    sample.frames[1]




.. parsed-literal::

    <FrameView: {
        'id': '636cfe832c777ac161910233',
        'frame_number': 1,
        'detections': <Detections: {
            'detections': BaseList([
                <Detection: {
                    'id': '636cfe82dd3cdcdcd97efbb3',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'person',
                    'bounding_box': BaseList([0.2525, 0.8288, 0.1812, 0.1678]),
                    'mask': None,
                    'confidence': 1.0,
                    'index': None,
                }>,
                <Detection: {
                    'id': '636cfe82dd3cdcdcd97efbb4',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'person',
                    'bounding_box': BaseList([0.63635, 0.00874999999999998, 0.1207, 0.3149]),
                    'mask': None,
                    'confidence': 1.0,
                    'index': None,
                }>,
                <Detection: {
                    'id': '636cfe82dd3cdcdcd97efbb5',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'person',
                    'bounding_box': BaseList([
                        0.30820000000000003,
                        0.32125000000000004,
                        0.1828,
                        0.5125,
                    ]),
                    'mask': None,
                    'confidence': 1.0,
                    'index': None,
                }>,
                <Detection: {
                    'id': '636cfe82dd3cdcdcd97efbb6',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'person',
                    'bounding_box': BaseList([0.5392, 0.7257, 0.2042, 0.2812]),
                    'mask': None,
                    'confidence': 1.0,
                    'index': None,
                }>,
                <Detection: {
                    'id': '636cfe82dd3cdcdcd97efbb7',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'sports ball',
                    'bounding_box': BaseList([
                        0.045313000000000006,
                        0.37777800000000006,
                        0.160156,
                        0.2375,
                    ]),
                    'mask': None,
                    'confidence': 1.0,
                    'index': None,
                }>,
                <Detection: {
                    'id': '636cfe82dd3cdcdcd97efbb8',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'sports ball',
                    'bounding_box': BaseList([
                        0.142969,
                        0.020833499999999998,
                        0.03125,
                        0.061111,
                    ]),
                    'mask': None,
                    'confidence': 1.0,
                    'index': None,
                }>,
                <Detection: {
                    'id': '636cfe82dd3cdcdcd97efbb9',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'chair',
                    'bounding_box': BaseList([
                        0.11015650000000002,
                        0.002777500000000002,
                        0.096875,
                        0.176389,
                    ]),
                    'mask': None,
                    'confidence': 1.0,
                    'index': None,
                }>,
                <Detection: {
                    'id': '636cfe82dd3cdcdcd97efbba',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'chair',
                    'bounding_box': BaseList([
                        0.18125000000000002,
                        0.0041665000000000035,
                        0.089062,
                        0.141667,
                    ]),
                    'mask': None,
                    'confidence': 1.0,
                    'index': None,
                }>,
                <Detection: {
                    'id': '636cfe82dd3cdcdcd97efbbb',
                    'attributes': BaseDict({}),
                    'tags': BaseList([]),
                    'label': 'chair',
                    'bounding_box': BaseList([
                        0.2460935,
                        0.0013889999999999944,
                        0.082031,
                        0.115278,
                    ]),
                    'mask': None,
                    'confidence': 1.0,
                    'index': None,
                }>,
            ]),
        }>,
    }>



Start reading the video file with OpenCV:

.. code:: ipython3

    vid=cv2.VideoCapture(sample.filepath)

.. code:: ipython3

    print("number of frames:",int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))


.. code-block:: text

    number of frames: 501


Let’s define a small helper function:

.. code:: ipython3

    def draw_detections(sample: fo.Sample, vid: cv2.VideoCapture, nframe: int):
        nmax=int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if nframe > nmax:
            raise AssertionError("max frame is " + str(nmax))
        ok = vid.set(cv2.CAP_PROP_POS_FRAMES, nframe-1)
        if not ok:
            raise AssertionError("seek failed")
        ok, arr = vid.read() # BGR image in arr
        if not ok:
            raise AssertionError("no image")
        for detection in sample.frames[nframe].detections.detections:
            x0, y0, w, h = detection.bounding_box # rel coords
            x1, y1, x2, y2 = floor(x0*arr.shape[1]), floor(y0*arr.shape[0]), floor((x0+w)*arr.shape[1]), floor((y0+h)*arr.shape[0])
            arr=cv2.rectangle(arr, (x1, y1), (x2, y2), (255, 0, 0), 5)
        return arr

.. code:: ipython3

    img=draw_detections(sample, vid, 200)
    img_ = img[:,:,::-1] # BGR -> RGB

.. code:: ipython3

    plt.imshow(img_)
    vid.release()



.. image:: cli_tutorial_7_nb_files/cli_tutorial_7_nb_22_0.png


For now, let’s get back to terminal command line.

Everything that you learned for image datasets, applies for video
datasets as well: ``compressai-vision import-custom`` can be used to
import mpeg-vcm datasets. ``compressai-vision app`` can be used to
visualize interactively datasets.

When using the fiftyone app, there is a small catch though. Web-browsers
are picky on the type of video they can play. For some video datasets,
in order to view them in the app, you need to create separate
“side-data” videos for visualization. These you can generate these
automagically with the ``compressai-vision make-thumbnails`` command.
Note that ``compressai-vision import-custom`` generates you these
thumbnails on-the-go when you import new video sets. Switching between
the main video and “side-data” video is demoed in `this
animation <https://voxel51.com/docs/fiftyone/_images/app-multiple-media-fields.gif>`__

In chapters 3 and 4 you learned how to evaluate models (in serial and
parallel) with the ``compressai-vision detectron2-eval`` command.

The same command can be used to evaluate video datasets as well. Here
the parameter ``--slice`` refers to videos, not individual image (as
usual, for a production run, you would remove the ``--slice``
parameter):

.. code:: bash

    compressai-vision detectron2-eval --y --dataset-name=sfu-hw-objects-v1 \
    --slice=1:2 \
    --scale=100 \
    --progressbar \
    --output=detectron2_test.json \
    --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml


.. code-block:: text

    importing fiftyone
    fiftyone imported
    WARNING: using a dataset slice instead of full dataset: SURE YOU WANT THIS?
    
    Using dataset          : sfu-hw-objects-v1
    Dataset media type     : video
    Dataset tmp clone      : detectron-run-sampsa-sfu-hw-objects-v1-2022-11-10-15-37-24-746313
    Image scaling          : 100
    WARNING: Using slice   : 1:2
    Number of samples      : 1
    Torch device           : cpu
    Detectron2 model       : COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
    Model was trained with : coco_2017_train
    ** Evaluation without Encoding/Decoding **
    Ground truth data field name
                           : detections
    Eval. results will be saved to datafield
                           : detectron-predictions
    Evaluation protocol    : open-images
    Progressbar            : True
    WARNING: progressbar enabled --> disabling normal progress print
    Print progress         : 0
    Output file            : detectron2_test.json
    Peek model classes     :
    ['airplane', 'apple', 'backpack', 'banana', 'baseball bat'] ...
    Peek dataset classes   :
    ['chair', 'person', 'sports ball'] ...
    cloning dataset sfu-hw-objects-v1 to detectron-run-sampsa-sfu-hw-objects-v1-2022-11-10-15-37-24-746313
    instantiating Detectron2 predictor
    USING VIDEO /home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1/ClassX/Annotations/BasketballDrill/video.mp4
    seeking to 2
    /home/sampsa/silo/interdigital/venv_all/lib/python3.8/site-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
    To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  ../aten/src/ATen/native/BinaryOps.cpp:467.)
      return torch.floor_divide(self, other)
     100% |███████████████████████████████████████████████████████████████████| 4/4 Evaluating detections...
     100% |███████████| 1/1 [71.8ms elapsed, 0s remaining, 13.9 samples/s] 
    deleting tmp database detectron-run-sampsa-sfu-hw-objects-v1-2022-11-10-15-37-24-746313
    
    Done!
    


Take a look at the results:

.. code:: ipython3

    cat detectron2_test.json


.. code-block:: text

    {
      "dataset": "sfu-hw-objects-v1",
      "gt_field": "detections",
      "tmp datasetname": "detectron-run-sampsa-sfu-hw-objects-v1-2022-11-10-15-37-24-746313",
      "slice": "1:2",
      "model": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
      "codec": "",
      "qpars": null,
      "bpp": [
        null
      ],
      "map": [
        0.5370370370370371
      ],
      "map_per_class": [
        {
          "chair": 0.1111111111111111,
          "person": 1.0,
          "sports ball": 0.5
        }
      ]
    }

