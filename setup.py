from pathlib import Path

from setuptools import find_packages, setup

# The following line is modified by setver.bash
version = "0.0.0"

this_folder = Path(__file__).resolve().parent

path = this_folder / "requirements.txt"
install_requires = []  # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if path.is_file():
    with path.open("r") as f:
        install_requires = f.read().splitlines()

TEST_REQUIRES = ["pytest", "pytest-cov"]
DEV_REQUIRES = TEST_REQUIRES + [
    "black",
    "flake8",
    "flake8-bugbear",
    "flake8-comprehensions",
    "isort",
    "mypy",
]


def get_extra_requirements():
    extras_require = {
        "dev": DEV_REQUIRES,
        "doc": ["sphinx", "furo"],
    }
    extras_require["all"] = {req for reqs in extras_require.values() for req in reqs}
    return extras_require


# # https://setuptools.readthedocs.io/en/latest/setuptools.html#basic-use
setup(
    name="compressai-vision",
    version=version,
    # install_requires = [
    #    "PyYAML",
    #    'docutils>=0.3', # # List here the required packages
    # ],
    install_requires=install_requires,  # instead, read from a file (see above)
    packages=find_packages(),  # # includes python code from every directory that has an "__init__.py" file in it.  If no "__init__.py" is found, the directory is omitted.  Other directories / files to be included, are defined in the MANIFEST.in file
    include_package_data=True,  # # conclusion: NEVER forget this : files get included but not installed
    # # "package_data" keyword is a practical joke: use MANIFEST.in instead
    # # WARNING: If you are using namespace packages, automatic package finding does not work, so use this:
    # packages=[
    #    'compressai_vision.subpackage1'
    # ],
    # scripts=[
    #    "bin/somescript"
    # ],
    # # "entry points" get installed into $HOME/.local/bin
    # # https://unix.stackexchange.com/questions/316765/which-distributions-have-home-local-bin-in-path
    entry_points={
        "console_scripts": [
            "compressai-vision = compressai_vision.cli.main:main",
            "compressai-nokia-auto-import = compressai_vision.cli.auto:main",
        ]
    },
    # metadata for upload to PyPI
    author="Sampsa Riikonen",
    author_email="sampsa.riikonen@iki.fi",
    description="Evaluation pipelines for Video Compression for Machine Vision on top of CompressAI",
    extras_require=get_extra_requirements(),
    license="BSD 3-Clause Clear License",
    # keywords="compressai",
    # url              = "nada", # project homepage
    long_description="""
    Evaluation pipeline for CompressAI
    """,
    long_description_content_type="text/plain",
    # long_description_content_type='text/x-rst', # this works
    # long_description_content_type='text/markdown', # this does not work
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        # 'Topic :: Multimedia :: Video', # set a topic
        # Pick your license as you wish
        # https://autopilot-docs.readthedocs.io/en/latest/license_list.html
        # 'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    # project_urls={ # some additional urls
    #    'Tutorial': 'nada
    # },
)
