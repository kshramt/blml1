import os

from setuptools import setup


def getversion():
    head = '__version__ = "'
    tail = '"\n'
    with open(os.path.join("blml1", "__init__.py")) as fp:
        for l in fp:
            if l.startswith(head) and l.endswith(tail):
                return l[len(head) : -len(tail)]
    raise Exception("__version__ not found")


setup(
    name="blml1",
    version=getversion(),
    description="blml1",
    url="https://github.com/kshramt/blml1",
    author="kshramt",
    packages=["blml1", "blml1._common"],
    install_requires=[],
    extras_require=dict(
        dev=["mypy", "pyflakes", "black", "pylint", "wheel", "twine", "pytype"]
    ),
    classifiers=["License :: OSI Approved :: GNU General Public License v3 (GPLv3)"],
    data_files=[(".", ["LICENSE.txt"])],
    zip_safe=True,
)
