import setuptools
import versioneer
new_version='0.1.4'
# conda install pytorch

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     install_requires=['networkx==1.11','matplotlib==2.2.3','pgmpy','numpy','pandas','tqdm','sklearn','funcsigs','statsmodels','community'],
#     install_requires=['networkx','matplotlib','pgmpy','numpy','pandas','tqdm','sklearn','funcsigs','statsmodels','community'],
     python_requires='>=3',
     name='bnlearn',
     version=new_version,
#     version=versioneer.get_version(),    # VERSION CONTROL
#     cmdclass=versioneer.get_cmdclass(),  # VERSION CONTROL
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="Python package for learning the graphical structure of Bayesian networks, parameter learning, inference and sampling methods.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/erdogant/bnlearn",
	 download_url = 'https://github.com/erdogant/bnlearn/archive/'+new_version+'.tar.gz',
     packages=setuptools.find_packages(), # Searches throughout all dirs for files to include
     include_package_data=True, # Must be true to include files depicted in MANIFEST.in
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
