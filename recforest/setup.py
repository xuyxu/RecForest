import os
import numpy
from distutils.version import LooseVersion
from numpy.distutils.misc_util import Configuration


CYTHON_MIN_VERSION = "0.24"


def configuration(parent_package="", top_path=None):

    libraries = []
    if os.name == "posix":
        libraries.append("m")

    config = Configuration("recforest", parent_package, top_path)

    config.add_extension("_transform",
                         sources=["_transform.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])

    message = ("Please install cython with a version >= {0} in order "
               "to build a recforest development version.").format(
               CYTHON_MIN_VERSION)
    try:
        import Cython
        if LooseVersion(Cython.__version__) < CYTHON_MIN_VERSION:
            message += " Your version of Cython was {0}.".format(
                Cython.__version__)
            raise ValueError(message)
        from Cython.Build import cythonize
    except ImportError as exc:
        exc.args += (message,)
        raise
    config.ext_modules = cythonize(config.ext_modules)

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
