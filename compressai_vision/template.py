"""
NAME.py : Description of the file

* Copyright: 2017 [copyright holder]
* Authors  : Sampsa Riikonen
* Date     : 2017
* Version  : 0.1

This file is part of the compressai_vision library

[copy-paste your license here]
"""
import sys


def test1():
    st = """Empty test
  """
    pre = __name__ + "test1 :"
    print(pre, st)


def test2():
    st = """Empty test
  """
    pre = __name__ + "test2 :"
    print(pre, st)


def main():
    pre = "main :"
    print(pre, "main: arguments: ", sys.argv)
    if (len(sys.argv) < 2):
        print(pre, "main: needs test number")
    else:
        st = "test" + str(sys.argv[1]) + "()"
        exec(st)


if (__name__ == "__main__"):
    main()

