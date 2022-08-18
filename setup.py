from setuptools import setup, Extension, find_packages
import sys, os

# The following line is modified by setver.bash
version = '0.0.0'

this_folder = os.path.dirname(os.path.realpath(__file__))
path = this_folder + '/requirements.txt'
install_requires = [] # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(path):
    with open(path) as f:
        install_requires = f.read().splitlines()

# # https://setuptools.readthedocs.io/en/latest/setuptools.html#basic-use
setup(
    name = "compressai-vision",
    version = version,
    #install_requires = [
    #    "PyYAML",
    #    'docutils>=0.3', # # List here the required packages
    #],

    install_requires = install_requires, # instead, read from a file (see above)

    packages = find_packages(), # # includes python code from every directory that has an "__init__.py" file in it.  If no "__init__.py" is found, the directory is omitted.  Other directories / files to be included, are defined in the MANIFEST.in file
    
    include_package_data=True, # # conclusion: NEVER forget this : files get included but not installed
    # # "package_data" keyword is a practical joke: use MANIFEST.in instead
    
    # # WARNING: If you are using namespace packages, automatic package finding does not work, so use this:
    #packages=[
    #    'compressai_vision.subpackage1'
    #],
    
    #scripts=[
    #    "bin/somescript"
    #],

    # # "entry points" get installed into $HOME/.local/bin
    # # https://unix.stackexchange.com/questions/316765/which-distributions-have-home-local-bin-in-path
    entry_points={
        'console_scripts': [
            'compressai-vision = compressai_vision.cli.main:main',
            'compressai-nokia-auto-import = compressai_vision.cli.auto:main'
        ]
    },
    
    # metadata for upload to PyPI
    author           = "Sampsa Riikonen",
    author_email     = "sampsa.riikonen@iki.fi",
    description      = "Evaluation pipeline for CompressAI",
    license          = "MIT",
    keywords         = "compressai",
    # url              = "nada", # project homepage
    
    long_description ="""
    Evaluation pipeline for CompressAI
    """,
    long_description_content_type='text/plain',
    # long_description_content_type='text/x-rst', # this works
    # long_description_content_type='text/markdown', # this does not work
    
    classifiers      =[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Operating System :: POSIX :: Linux',
        # 'Topic :: Multimedia :: Video', # set a topic
        # Pick your license as you wish
        # https://autopilot-docs.readthedocs.io/en/latest/license_list.html
        # 'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.4',
        #'Programming Language :: Python :: 3.5',
        #'Programming Language :: Python :: 3.6',
    ],
    #project_urls={ # some additional urls
    #    'Tutorial': 'nada
    #},
)
