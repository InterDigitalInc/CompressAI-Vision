These bash scripts can be run to test the CLI

Clone and build VTM (keeping default build output folders) and inform the CLI about the root folder of the code.

For custom model compression, use your own model and checkpoints, or use default compressai and download pre-trained checkpoints to the same folder as your model.py.


Example of basic test:
```
bash scripts/tests/runall.bash -v ~/vvc/vtm-12.0
```

Example of folder for test 06:
examples/models/bmshj2018-factorized/
- model.py
- bmshj2018-factorized-prior-1-446d5c7f.pth.tar
- bmshj2018-factorized-prior-2-87279a02.pth.tar

download checkpoints using e.g.
```
curl https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-2-87279a02.pth.tar -o bmshj2018-factorized-prior-2-87279a02.pth.tar
```
