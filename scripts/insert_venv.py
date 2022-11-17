import sys, os
from importlib.metadata import version
fo_version=version("fiftyone").replace(".","-")
if sys.prefix == sys.base_prefix:
    print("WARNING: youre NOT in a virtualenv")
    sys.exit(2)
with open(os.path.join(sys.prefix,"bin","activate"), "a") as f:
    f.write('export FIFTYONE_DATABASE_NAME="fiftyone-'+fo_version+'"\n')
