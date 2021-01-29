import os
import sys
from setuptools import find_packages
from numpy.distutils.core import setup


DISTNAME = "RecForest"
MAINTAINER = "Yi-Xuan Xu"
MAINTAINER_EMAIL = "xuyx@lamda.nju.edu.cn"
DESCRIPTION = ("Implementation of Reconstruction-based Anomaly Detection with"
               " Completely Random Forest")
LICENSE = "BSD 3-Clause"
URL = "https://github.com/xuyxu/RecForest"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/xuyxu/RecForest/issues",
    "Documentation": "https://github.com/xuyxu/RecForest/README.rst",
    "Source Code": "https://github.com/xuyxu/RecForest"}
VERSION = "0.1.0"
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.add_subpackage("recforest")

    return config


if __name__ == "__main__":

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          project_urls=PROJECT_URLS,
          version=VERSION,
          long_description=LONG_DESCRIPTION,
          classifiers=[
              "Intended Audience :: Science/Research",
              "Intended Audience :: Developers",
              "Topic :: Software Development",
              "Topic :: Scientific/Engineering",
              "Operating System :: Microsoft :: Windows",
              "Operating System :: POSIX",
              "Operating System :: Unix",
              "Programming Language :: Python :: 3",
              "Programming Language :: Python :: 3.6",
              "Programming Language :: Python :: 3.7",
              "Programming Language :: Python :: 3.8"],
          keywords=["Anomaly Detection", "Decition Tree Ensemble"],
          python_requires=">=3.6",
          install_requires=[
              "numpy>=1.13.3",
              "scipy>=0.19.1",
              "joblib>=0.12",
              "cython>=0.28.5",
              "scikit-learn>=0.22",
          ],
          packages=find_packages("recforest", exclude=["examples"]),
          include_package_data=True,
          zip_safe=False,
          setup_requires=["cython"])
