
FAQ
===

1. Error: "failed to bind"
--------------------------

You can get this error message if fiftyone has a problem in connecting
to the mongodb server:

.. code-block:: text

    fiftyone.core.service.ServiceListenTimeout: fiftyone.core.service.DatabaseService failed to bind to port

If you haven't defined an external mongodb server, each time the python code imports
fiftyone for the first time, an "internal" mongodb server instance is started - on some occasions that internal mongodb 
server might have exited in a "dirty" manner.

No need to panic.  We provide a command-line tool for cleaning things up.  Just type:

.. code-block:: bash

    compressai-vision mongo stop

2. Error: "you must have fiftyone>x.x.x installed"
--------------------------------------------------

If you get an error of this kind:

.. code-block:: text

    You must have fiftyone>=0.17.2 installed in order to migrate from v0.17.2 to v0.16.6, but you are currently running fiftyone==0.16.6.

First of all, make sure that you are **not running a mongodb server on your linux box**, as fiftyone starts its own (internal/bundled) mongodb server instance!

The above error occurs typically, when you have created a database with a certain version of fiftyone (say, v0.16.6), but then you have (or someone else has) 
touched it with a different version of fiftyone (for example 0.17.2). 

A nice fix to this one is to set a unique database name according to the fiftyone version you are using.  

As described `here <https://voxel51.com/docs/fiftyone/user_guide/config.html#configuration-options>`_, the database name is set with the ``FIFTYONE_DATABASE_NAME`` environmental
variable, so set in your virtualenv (in ``bin/activate``) ``export FIFTYONE_DATABASE_NAME=fiftyone-0.16.0`` (or whatever your version number might be).

To wipe out the (incompatible) database, you can always do this:

.. code-block:: bash

    compressai-vision mongo clear

