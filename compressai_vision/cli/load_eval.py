"""cli load_eval functionality
"""
import pickle


def main(p):
    dataset = fo.load_dataset(p.name)
    print("removing dataset %s from fiftyone" % (p.name))
    if not p.y:
        input("press enter to continue.. ")
    try:
        fo.delete_dataset(p.name)
    except ValueError as e:
        print("could not deregister because of", e)
