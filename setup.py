import setuptools
from os import path

from RL_for_reco import __version__

here = path.abspath(path.dirname(__file__))

requires_list = [
        "thrift-sasl==0.3.0", 
        "hvac>=0.9.6",
        "pyhive[hive]",
        "pyarrow>=0.16.0",
        "pandas==1.0.3",
        "slackclient>=2.5.0",
        "google-cloud-bigquery>=1.24.0",
        "httplib2>=0.18.0",
        "click",
        "PyGithub",
        "pycryptodome",
        "joblib==0.14.1",
        "tabulate>=0.8.7",
]

air = [
    "torch==1.4.0",
    "torchvision==0.5.0",
    "scikit-learn==0.22.2.post1",
    "scipy==1.4.1",
    "matplotlib==3.2.1",
    "mushroom_rl==1.4.0",
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
    install_requires=requires_list,
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 ],
    extras_require={"air": air},
)