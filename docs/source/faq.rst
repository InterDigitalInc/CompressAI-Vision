
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

2. Error: "you must have fiftyone>x.x.x installed"
--------------------------------------------------

If you get an error of this kind:

::

    You must have fiftyone>=0.17.2 installed in order to migrate from v0.17.2 to v0.16.6, but you are currently running fiftyone==0.16.6.

Each version of fiftyone uses a certain version of mongodb.  The above error raises typically, when you have created the database with a certain
version of fiftyone (say, v0.16.6), but then try to access it with a different version of fiftyone (for example 0.17.2). 

A typical confusion is that you use two different versions of fiftyone, each from a different virtualenv.

Please use *only* the same virtualenv from where you created the database in the first place.  

To wipe out the (incompatible) database, you can always do this:

::

    compressai-vision-mongo clear

