import sys, os
from importlib.metadata import version
fo_version=version("fiftyone").replace(".","-")
if sys.prefix == sys.base_prefix:
    print("WARNING: youre NOT in a virtualenv")
    sys.exit(2)

# val=os.environ["USER"]+"-fiftyone-"+fo_version # nopes.. it's a better idea that users create personal copies of imported datasets
val="fiftyone-"+fo_version
print("patching environmental variables with")
print("FIFTYONE_DATABASE_NAME="+val)

with open(os.path.join(sys.prefix,"bin","activate"), "a") as f:
    f.write("export FIFTYONE_DATABASE_NAME="+val+"\n")
