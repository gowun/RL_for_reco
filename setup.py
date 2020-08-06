import setuptools
from os import path

from RL_for_reco import __version__

here = path.abspath(path.dirname(__file__))

air = [
    "torch>=1.4.0",
    "torchvision>=0.5.0",
    "scikit-learn>=0.22.2.post1",
    "scipy>=1.4.1",
    "matplotlib>=3.2.1",
    "mushroom_rl>=1.4.0",
]


setuptools.setup(
    name='RL_for_reco',
    version=__version__,
    description='A Python toolkit of Deep Reinforcement Learning for Structured Data-Oriented Recommendation.',
    url='https://github.com/gowun/RL_for_reco.git',
    author="Gowun Jeong",
    author_email='gowun.jeong@gmail.com',
    license='MIT',
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    zip_safe=False,
    long_description=open('README.md').read(),
    install_requires=air,
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 ],
)
