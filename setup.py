from setuptools import Command, find_packages, setup

__lib_name__ = "SpaKnit"
__lib_version__ = "1.0.0"
__description__ = "SpaKnit is a multimodal integration framework for spatial multi-omics data."
__url__ = "https://github.com/XJTU-MingxuanLi/SpaKnit"
__author__ = "Mingxuan Li"
__author_email__ = "3123154029@stu.xjtu.edu.cn"
__license__ = "MIT"
__keywords__ = ["Spatial multi-omics", "Implicit neural representation", "Deep Canonically Correlated Auto-Encoder"]

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

from pkg_resources import parse_requirements
with open("requirements.txt", encoding="utf-8") as fp:
    install_requires = [str(requirement) for requirement in parse_requirements(fp)]

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages=find_packages(),
    install_requires=install_requires,
    zip_safe = False,
    include_package_data = True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
