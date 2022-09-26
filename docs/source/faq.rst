
FAQ
===

1. Error: "failed to bind"
--------------------------

You can get this error message if fiftyone has a problem in connecting
to the mongodb server:

::

    fiftyone.core.service.ServiceListenTimeout: fiftyone.core.service.DatabaseService failed to bind to port

If you haven't defined an external mongodb server, each time the python code imports
fiftyone for the first time, an "internal" mongodb server instance is started - on some occasions that internal mongodb 
server might have exited in a "dirty" manner.

No need to panic.  We provide a command-line tool for cleaning things up.  Just type:

::

    compressai-vision-mongo

and follow the instruction

