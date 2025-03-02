import setuptools
import re

# versioning ------------
VERSIONFILE="bnlearn/__init__.py"
getversion = re.search( r"^__version__ = ['\"]([^'\"]*)['\"]", open(VERSIONFILE, "rt").read(), re.M)
if getversion:
    new_version = getversion.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# Setup ------------
with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()
setuptools.setup(
     install_requires=["pgmpy>=0.1.18",
                       "networkx>=2.7.1",
                       "matplotlib>=3.3.4",
                       "numpy>=1.24.1",
                       'pandas',
                       'tqdm',
                       'ismember',
                       'scikit-learn',
                       'funcsigs',
                       'statsmodels',
                       'python-louvain',
                       'packaging',
                       'df2onehot',
                       'fsspec',
                       'pypickle',
                       'tabulate',
                       'ipywidgets',
                       'datazets',
                       'setgraphviz',
                       'lingam'],
     python_requires='>=3',
     name='bnlearn',
     version=new_version,
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="Python package for Causal Discovery by learning the graphical structure of Bayesian networks, parameter learning, inference and sampling methods.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://erdogant.github.io/bnlearn",
	 download_url = 'https://github.com/erdogant/bnlearn/archive/'+new_version+'.tar.gz',
     packages=setuptools.find_packages(), # Searches throughout all dirs for files to include
     include_package_data=True, # Must be true to include files depicted in MANIFEST.in
     license_files=["LICENSE"],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
