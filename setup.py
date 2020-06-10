from setuptools import setup, find_packages
from codecs import open
from os import path

from RL_for_reco import __version__

here = path.abspath(path.dirname(__file__))

requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))


setup(
    name='RL_for_reco',
    version=__version__,
    description='A Python toolkit of Deep Reinforcement Learning for Structured Data-Oriented Recommendation.',
    url='https://github.com/gowun/RL_for_reco',
    author="Gowun Jeong",
    author_email='gowun.jeong@gmail.com',
    license='MIT',
    packages=[package for package in find_packages()
              if package.startswith('RL_for_reco')],
    zip_safe=False,
    long_description=open('README.md').read(),
    install_requires=requires_list,
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 ]
)