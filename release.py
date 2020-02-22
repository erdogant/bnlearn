"""Make new release on github and pypi."""
# --------------------------------------------------
# Name        : irelease.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/irelease
# Licence     : MIT
# --------------------------------------------------

import os
import re
import platform
import argparse
import numpy as np
import urllib.request
import shutil
from packaging import version
import webbrowser
EXCLUDE_DIR = np.array(['depricated','__pycache__','_version','.git','.gitignore','build','dist','doc','docs'])


# %% Make executable:
def make_script():
    """Create bash file to release your package.

    Returns
    -------
    release.sh
    release.py

    """
    py_path = os.path.dirname(os.path.abspath(__file__))
    release_path = os.path.join(py_path, 'irelease.py')
    f = open("release.sh", "w")
    f.write('#!/bin/sh')
    f.write('\necho "release your packge.."')
    f.write('\npython "' + release_path + '"')
    f.write('\nread -p "Press [Enter] to close"')
    f.close()
    # Copy irelease script
    shutil.copyfile(release_path, os.path.join(os.getcwd(), 'release.py'))
    print('[irelease] release.sh created and .py file copied!')


# %% def main(username, packagename=None, verbose=3):
def main(username, packagename, clean=False, twine=None, verbose=3):
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

    Parameters
    ----------
    username : str
        Name of the github account.
    packagename : str
        Name of the package.
    clean : bool
        Clean local distribution files for packaging.
    twine : str
        Filepath to the executable of twine.
    verbose : int
        Print message. The default is 3.

    Returns
    -------
    None.


    References
    ----------
    * https://dzone.com/articles/executable-package-pip-install
    * https://blog.ionelmc.ro/presentations/packaging/#slide:8

    """
    # Set defaults
    username, packagename, clean, twine, verbose = _set_defaults(username, packagename, clean, twine, verbose)
    # Get package name
    packagename = _package_name(packagename, verbose=verbose)
    if packagename is None: raise Exception('[irelease] ERROR: Package directory does not exists.')
    if username is None: raise Exception('[irelease] ERROR: Github name does not exists.')

    # Get init file from the dir of interest
    initfile = os.path.join(packagename, "__init__.py")

    if verbose>=3:
        os.system('cls')
        print('----------------------------------')
        print('[irelease] username  : %s' %username)
        print('[irelease] Package   : %s' %packagename)
        print('[irelease] Cleaning  : %s' %clean)
        print('[irelease] Verbosity : %s' %verbose)
        print('[irelease] init file : %s' %initfile)

    # Find version
    if os.path.isfile(initfile):
        # Extract version from __init__.py
        getversion = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", open(initfile, "rt").read(), re.M)
        if getversion:
            # Remove build directories
            if verbose>=3 and clean:
                input("[irelease] Press [Enter] to clean previous local builds from the package directory..")
                _make_clean(packagename, verbose=verbose)
            # Version found, lets move on:
            current_version = getversion.group(1)
            # Get latest version of github release
            githubversion = github_version(username, packagename, verbose=verbose)

            # Print info about the version
            if githubversion=='0.0.0':
                if verbose>=3: print("[irelease] Very first release for [%s]" %(packagename))
                VERSION_OK = True
            elif githubversion=='9.9.9':
                if verbose>=3: print("[irelease] %s/%s not available at github." %(username, packagename))
                VERSION_OK = False
            elif version.parse(current_version)>version.parse(githubversion):
                if verbose>=3: print('[irelease] Current local version from __init__.py: %s and from github: %s' %(current_version, githubversion))
                VERSION_OK = True
            else:
                VERSION_OK = False

            # Continue is version is TRUE
            if VERSION_OK:
                if verbose>=3: input("[irelease] Press Enter to make build and release [%s] on github..." %(current_version))
                # Make build and install
                _make_build_and_install(packagename, current_version)
                # Set tag to github and push
                _github_set_tag_and_push(current_version, verbose=verbose)
                # Upload to pypi
                if os.path.isfile(twine):
                    if verbose>=3: input("[irelease] Press Enter to upload to pypi...")
                    twine_line = twine + ' upload dist/*'
                    if verbose>=3: print('[irelease] %s' %(twine_line))
                    os.system(twine_line)
                # Fin message and webbrowser
                _fin_message(username, packagename, current_version, githubversion, verbose)

            elif (githubversion != '9.9.9') and (githubversion != '0.0.0'):
                if verbose>=2: print('[irelease] WARNING: Not released! You need to increase your version: [%s]' %(initfile))
                if verbose>=2: print('[irelease] WARNING: Local version : %s' %(current_version))
                if verbose>=2: print('[irelease] WARNING: github version: %s' %(githubversion))

        else:
            if verbose>=1: print("[irelease] ERROR: Unable to find version string in %s. Make sure that the operators are space seperated eg.: __version__ = '0.1.0'" % (initfile,))
    else:
        if verbose>=2: print('[irelease] Warning: __init__.py File not found: %s' %(initfile))

    if verbose>=3: input("[irelease] Press [Enter] to exit.")


# %% Final message
def _fin_message(username, packagename, current_version, githubversion, verbose):
    if verbose>=2: print('[irelease] ==================================================================')
    if verbose>=2: print('[irelease] FIN!')
    if verbose>=2: print('[irelease] >  But you still need to do one more thing.')
    if verbose>=2: print('[irelease] 1. Go to your github most recent releases.')
    if verbose>=2: print('[irelease] 2. Press [edit tag]')
    if verbose>=2: print('[irelease] 3. Set version in [Release title] to: %s' %(current_version))
    if verbose>=2: print('[irelease] ==================================================================')

    # Open webbroswer and navigate to github to add version
    if verbose>=3: input("[irelease] Press Enter to navigate...")
    git_release_link = 'https://github.com/' + username + '/' + packagename + '/releases/tag/' + current_version
    webbrowser.open(git_release_link, new=2)
    if verbose>=2: print('[irelease] %s' %(git_release_link))
    if verbose>=2: print('[irelease] ==================================================================')


# %% Get latest github version
def github_version(username, packagename, verbose=3):
    """Get latest github version for package.

    Parameters
    ----------
    username : String
        Name of the github account.
    packagename : String
        Name of the package.
    verbose : int, optional
        Print message. The default is 3.

    Returns
    -------
    github_version : String
        x.x.x. : Version number of the latest github package.
        0.0.0  : When repo has no tag/release yet.
        9.9.9  : When repo is private or package/user does not exists.

    """
    # Pull latest from github
    print('[irelease] git pull')
    os.system('git pull')

    # Check whether username/repo exists and not private
    try:
        github_page = None
        github_url = 'https://api.github.com/repos/' + username + '/' + packagename + '/releases'
        github_page = str(urllib.request.urlopen(github_url).read())
        tag_name = re.search('"tag_name"', github_page)
    except:
        if verbose>=1: print('[irelease] ERROR: github %s does not exists or is private.' %(github_url))
        github_version = '9.9.9'
        return github_version

    # Continue and check whether this is the very first tag/release or a multitude are readily there.
    if tag_name is None:
        if verbose>=4: print('[release.debug] github exists but tags and releases are empty [%s]' %(github_url))
        # Tag with 0.0.0 to indicate that this is a very first tag
        github_version = '0.0.0'
    else:
        # exists
        try:
            # Get the latest release
            github_url = 'https://api.github.com/repos/' + username + '/' + packagename + '/releases/latest'
            github_page = str(urllib.request.urlopen(github_url).read())
            tag_name = re.search('"tag_name"', github_page)
            # Find the next tag by the seperation of the comma. Do +20 or so to make sure a very very long version would also be included.
            # github_version = yaml.load(github_page)['tag_name']
            tag_ver = github_page[tag_name.end() + 1:(tag_name.end() + 20)]
            next_char = re.search(',', tag_ver)
            github_version = tag_ver[:next_char.start()].replace('"','')
        except:
            if verbose>=1:
                print('[irelease] ERROR: Can not find the latest github version!')
                print('[irelease] ERROR: Maybe repo Private or does not exists?')
            github_version = '9.9.9'

    if verbose>=4: print('[irelease] Github version: %s' %(github_version))
    if verbose>=4: print('[irelease] Github version requested from: %s' %(github_url))
    return github_version


def _make_build_and_install(packagename, current_version):
    # Make new build
    print('Making new wheel..')
    os.system('python setup.py bdist_wheel')
    # Make new build
    print('Making source build..')
    os.system('python setup.py sdist')
    # Install new wheel
    print('Installing new wheel..')
    os.system('pip install -U dist/' + packagename + '-' + current_version + '-py3-none-any.whl')


def _github_set_tag_and_push(current_version, verbose=3):
    # git commit
    if verbose>=3: print('[irelease] git add->commit->push')
    os.system('git add .')
    os.system('git commit -m ' + current_version)
    # os.system('git commit -m v' + current_version)
    # os.system('git push')
    # Set tag for this version
    if verbose>=3: print('Set new version tag: %s' %(current_version))
    # git tag -a 0.1.0 -d "0.1.0"
    # os.system('git tag -a v' + current_version + ' -m "v' + current_version + '"')
    os.system('git tag -a ' + current_version + ' -m "' + current_version + '"')
    os.system('git push origin --tags')


def _make_clean(packagename, verbose=3):
    if verbose>=3: print('[irelease] Removing local build directories..')
    if os.path.isdir('dist'): shutil.rmtree('dist')
    if os.path.isdir('build'): shutil.rmtree('build')
    if os.path.isdir(packagename + '.egg-info'): shutil.rmtree(packagename + '.egg-info')


def _package_name(packagename, verbose=3):
    # Infer name of the package by excluding all known-required-files-and-folders.
    if packagename is None:
        if verbose>=4: print('[irelease] Infer name of the package from the directory..')
        # List all folders in dir
        filesindir = np.array(os.listdir())
        getdirs = filesindir[list(map(lambda x: os.path.isdir(x), filesindir))]
        # Remove all the known not relevant files and dirs
        Iloc = np.isin(np.array(list(map(str.lower, getdirs))), EXCLUDE_DIR)==False  # noqa
        if np.any(Iloc):
            packagename = getdirs[Iloc][0]

    if verbose>=4: print('[irelease] Done! Working on package: [%s]' %(packagename))
    return(packagename)


# %% Extract github username from config file
def _github_username(verbose=3):
    username=None
    if verbose>=4: print('[release.debug] Extracting github name from .git folder')
    # Open github config file
    f = open('./.git/config')
    gitconfig = f.readlines()
    # Iterate over the lines and search for git@github.com
    for line in gitconfig:
        line = line.replace('\t','')
        geturl = re.search('git@github.com', line)
        # If git@github.com detected: exract the username
        if geturl:
            username_line = line[geturl.end() + 1:(geturl.end() + 20)]
            next_char = re.search('/',username_line)
            username = username_line[:next_char.start()].replace('"','')

    return username


# %% Extract github username from config file
def _github_package(verbose=3):
    package = None
    if verbose>=4: print('[release.debug] Extracting package name from .git folder')
    # Open github config file
    f = open('./.git/config')
    gitconfig = f.readlines()
    # Iterate over the lines and search for git@github.com
    for line in gitconfig:
        line = line.replace('\t','')
        geturl = re.search('git@github.com', line)
        # If git@github.com detected: exract the package
        if geturl:
            repo_line = line[geturl.end():]
            start_pos = re.search('/',repo_line)
            stop_pos = re.search('.git',repo_line)
            package = repo_line[start_pos.end():stop_pos.start()].replace('"','')

    return package


def _set_defaults(username, packagename, clean, twine, verbose):
    # Default verbosity value is 0
    if verbose is None:
        verbose=3

    if clean is None or clean==1:
        clean=True
    else:
        clean=False

    if twine is None:
        twine = ''
        if platform.system().lower()=='windows':
            twine = os.environ['TWIN.EXE']
            # TWINE_PATH = 'C://Users/<USER>/AppData/Roaming/Python/Python36/Scripts/twine.exe'

    if username is None:
        username = _github_username(verbose=verbose)

    if packagename is None:
        packagename = _github_package(verbose=verbose)

    return username, packagename, clean, twine, verbose


# %% Main function
if __name__ == '__main__':
    # main
    parser = argparse.ArgumentParser()
    # parser.add_argument("github", type=str, help="github account name")
    parser.add_argument("-u", "--username", type=str, help="username github.")
    parser.add_argument("-p", "--package", type=str, help="Dir of the package to be released.")
    parser.add_argument("-c", "--clean", type=int, choices=[0, 1], help="Remove local builds: [dist], [build] and [x.egg-info] before creating new ones.")
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2, 3, 4, 5], help="Output verbosity, higher number tends to more information.")
    parser.add_argument("-t", "--twine", type=str, help="Path to twine that is used to upload to pypi.")
    args = parser.parse_args()

    # Go to main
    main(args.username, args.package, clean=args.clean, twine=args.twine, verbose=args.verbosity)
