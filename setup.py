"""Setup script for python packaging."""
import site
import sys

from setuptools import find_packages, setup

# enable installing package for user
# https://github.com/pypa/pip/issues/7953#issuecomment-645133255
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setup(
    name="LightMHC",
    version="0.1.0",
    description="LightMHC: A Light Model for pMHC Structure Prediction with Graph Neural Networks",
    author="InstaDeep Ltd",
    url="https://github.com/instadeepai/lightmhc",
    license="CC BY-NC-SA 4.0",
    packages=find_packages(),
    python_requires=">=3.8,<3.9",
    include_package_data=True,
    zip_safe=False,
)
