"""Make new release on github and pypi.

Description
-----------
A new release is created by taking the underneath steps:
    1. List all files in current directory and exclude all except the directory-of-interest
    2. Extract the version from the __init__.py file
    3. Remove old build directories such as dist, build and x.egg-info
    4. Git pull
    5. Get latest version from github
    6. Check if the current version is newer then github lates--version.
        a. Make new wheel, build and install package
        b. Set tag to newest version and push to git
        c. Upload to pypi (credentials required)
"""

import os
import re
import numpy as np
import urllib.request
import yaml
import shutil
from packaging import version
yaml.warnings({'YAMLLoadWarning': False})
# GITHUB
GITHUBNAME = 'erdogant'
TWINE_PATH = 'C://Users/Erdogan/AppData/Roaming/Python/Python36/Scripts/twine.exe upload dist/*'


def make_build_and_install(PACKAGE_NAME, current_version):
    # Make new build
    print('Making new wheel..')
    os.system('python setup.py bdist_wheel')
    # Make new build
    print('Making source build..')
    os.system('python setup.py sdist')
    # Install new wheel
    print('Installing new wheel..')
    os.system('pip install -U dist/' + PACKAGE_NAME + '-' + current_version + '-py3-none-any.whl')


def github_set_tag_and_push(current_version):
    # git commit
    print('git add->commit->push')
    os.system('git add .')
    os.system('git commit -m v'+current_version)
    # os.system('git push')
    # Set tag for this version
    print('Set new version tag: %s' %(current_version))
    os.system('git tag -a v' + current_version + ' -m "' + current_version + '"')
    os.system('git push origin --tags')


def github_latest_version(GITHUBNAME, PACKAGE_NAME, verbose=3):
    # Get latest version of github release
    try:
        github_url = 'https://api.github.com/repos/' + GITHUBNAME + '/' + PACKAGE_NAME + '/releases/latest'
        github_page = urllib.request.urlopen(github_url)
        github_data = github_page.read()
        github_version = yaml.load(github_data)['tag_name']
        if verbose>=3: print('Github version: %s' %(github_version))
        if verbose>=3: print('Github version requested from: %s' %(github_url))
    except:
        if verbose>=0: print('Error: Can not check version if Github is private [https://github.com//%s/%s]' %(GITHUBNAME, PACKAGE_NAME))
        github_version = '9.9.9'
    
    return github_version


def make_clean(PACKAGE_NAME, verbose=3):
    if verbose>=3: print('Removing local build directories..')
    if os.path.isdir('dist'): shutil.rmtree('dist')
    if os.path.isdir('build'): shutil.rmtree('build')
    if os.path.isdir(PACKAGE_NAME + '.egg-info'): shutil.rmtree(PACKAGE_NAME + '.egg-info')


# %% Main function
if __name__ == '__main__':
    # Clean screen
    os.system('cls')
    # List all files in dir
    filesindir = np.array(list(map(lambda x: x.lower(), np.array(os.listdir()))))
    # Remove all the known not relevant files and dirs
    exclude = np.array(['depricated','__pycache__','_version','make_new_release.py','get_release_github.sh','setup.py','get_version.py','readme.md','.git','.gitignore','build','dist','docs','license','make_new_build.sh','make_new_realease.sh','manifest.in','requirements.txt','setup.cfg','test.py'])  # noqa
    PACKAGE_NAME = filesindir[np.isin(filesindir, exclude)==False][0]  # noqa
    # This must be the dir of interest
    INIT_FILE = PACKAGE_NAME + "/__init__.py"
    print('Working on package: [%s]' %(PACKAGE_NAME))

    # Find version now
    if os.path.isfile(INIT_FILE):
        # Extract version from __init__.py
        getversion = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", open(INIT_FILE, "rt").read(), re.M)
        if getversion:
            # Remove build directories
            make_clean(PACKAGE_NAME, verbose=3)
            # Pull latest from github
            print('git pull')
            os.system('git pull')
            # This is the current version described in our __init__.py file
            current_version = getversion.group(1)
            print('Current version: %s' %(current_version))
            # Get latest version of github release
            github_version = github_latest_version(GITHUBNAME, PACKAGE_NAME, verbose=3)

            # Continue with the process of building a new version if the current version is really newer then the one on github!
            VERSION_OK = version.parse(current_version)>version.parse(github_version)
            if VERSION_OK:
                input("Press Enter to continue...")
                # Make build and install
                make_build_and_install(PACKAGE_NAME, current_version)
                # Set tag to github and push
                github_set_tag_and_push(current_version)
                # Upload to pypi
                print('Upload to pypi..')
                os.system(TWINE_PATH)
            else:
                if github_version != '9.9.9':
                    print('Not released! You need to increase your version: [%s]' %(INIT_FILE))

        else:
            raise RuntimeError("Unable to find version string in %s." % (INIT_FILE,))
    else:
        print('Oh noo File not found: %s' %(INIT_FILE))
