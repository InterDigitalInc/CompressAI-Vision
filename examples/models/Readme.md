# Example of custom model for compressai-vision detectron2_eval

When you want to eval your model in the same conditions as other models and
state-of-the-art codecs supported in CompressAI-Vision, you can provide
your own architecture description within a model.py.

This Folder contains an examplary model.py exactly reproducing the model
"bmshj2018-factorized" from the CompressAI library.
For your custom model, you can inherit all things from compressai library.

You can test it by dowmloading a corresponding pre-trained model, e.g.
```bash
curl https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-1-446d5c7f.pth.tar -o bmshj2018-factorized-prior-1-446d5c7f.pth.tar
```

Then run e.g.
```bash
compressai-vision \
    detectron2-eval \
    --y \
    --dataset-name mpeg-vcm-detection \
    --slice 0:2 \
    --scale 100 \
    --progress 1 \
    --compression-model-path examples/models/bmshj2018-factorized \
    --compression-model-checkpoint \
        examples/models/bmshj2018-factorized/bmshj2018-factorized-prior-1-446d5c7f.pth.tar \
        examples/models/bmshj2018-factorized/bmshj2018-factorized-prior-2-87279a02.pth.tar \
    --output=detectron2_bmshj2018-factorized.json \
    --model=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
```