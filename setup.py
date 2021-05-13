from setuptools import find_packages, setup

# read the contents of README file
from os import path
from io import open  # for Python 2 and 3 compatibility

this_directory = path.abspath(path.dirname(__file__))

# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(name='DGFraud-TF2',
      version="0.1.0",
      author="Yingtong Dou, Zhongzheng Lu, Zhiqin Yang, Kay Liu, Yutong Deng, Hengrui Zhang, Zhiwei Liu and UIC BDSC Lab",
      author_email="bdscsafegraph@gmail.com",
      description='A Deep Graph-based Toolbox for Fraud Detection in Tensorflow 2.X',
      long_description=open("README.md", "r", encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      url='https://github.com/safe-graph/DGFraud-TF2',
      download_url='https://github.com/safe-graph/DGFraud-TF2/archive/master.zip',
      keywords=['fraud detection', 'anomaly detection', 'graph neural network',
                'data mining', 'security'],
      install_requires=['numpy>=1.16.4',
                        'tensorflow>=2.0',
                        'scipy>=1.2.1',
                        'scikit_learn>=0.21rc2',
                        'tqdm>=4.31.1'
                        ],
      packages=find_packages(exclude=['test']),
      include_package_data=True,
      setup_requires=['setuptools>=38.6.0'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Education',
          'Intended Audience :: Financial and Insurance Industry',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Intended Audience :: Information Technology',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
      ],
      )
