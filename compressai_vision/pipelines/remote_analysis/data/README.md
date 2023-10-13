 
You can access these files in your code without knowing the exact dir location with:

this:
```
from compressai_vision.tools import getDataFile

filename=getDataFile("README.md")
```
would return the absolute path of _this_ file.

[../../MANIFEST.in](../../MANIFEST.in) takes care that all files are included into the python package.

