"""
local.py : Local $HOME directory management

* Copyright: 2020 Sampsa Riikonen
* Authors  : Sampsa Riikonen
* Date     : 2020
* Version  : 0.1

This file is part of the python skeleton example library

Skeleton example library is free software: you can redistribute it and/or modify it under the terms of the MIT License.  
This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
See the MIT License for more details.   
"""
import sys
import os
import shutil

home = os.path.expanduser("~")

class LocalDir:
    basedir = ".some_hidden_dir"

    def __init__(self, *args):
        self.dirname = os.path.join(home, self.basedir, *args)
        self.make()

    def get(self):
        return self.dirname

    def reMake(self):
        self.clear()
        self.make()

    def make(self):
        if not self.has(self.dirname):
            os.makedirs(self.dirname)

    def clear(self):
        if self.dirname == home:
            raise BaseException("sanity exception - are you nuts!?")
            return
        try:
            shutil.rmtree(self.dirname)
        except Exception as e:
            pass

    def has(self, fname):
        return os.path.exists(os.path.join(self.dirname, fname))

    def getFile(self, fname):
        return os.path.join(self.dirname, fname)

    def getFileIf(self, fname):
        if self.has(fname):
            return self.getFile(fname)
        else:
            return None


class AppLocalDir(LocalDir):
    basedir = ".skeleton"


if __name__ == "__main__":
    config_dir = AppLocalDir("kokkelis")
    deeper_dir = AppLocalDir("kokkelis","fs")
    print(config_dir.getFile("diibadaaba"))
    config_dir.clear()
    deeper_dir.clear()

