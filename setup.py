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
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     install_requires=["pgmpy>=0.1.13", "networkx>2.5","matplotlib>=3.3.4",'numpy','pandas','tqdm','ismember','sklearn','funcsigs','statsmodels','community','packaging','wget','df2onehot','fsspec','pypickle','tabulate','ipywidgets', 'pyvis'],
     python_requires='>=3',
     name='bnlearn',
     version=new_version,
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="Python package for learning the graphical structure of Bayesian networks, parameter learning, inference and sampling methods.",
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
