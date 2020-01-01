echo "Cleaning previous builds first.."
rm -rf dist
rm -rf build
rm -rf hnet.egg-info

read -p "Making source build after pressing [Enter].."
echo "Making new build.."
echo ""
python setup.py bdist_wheel
echo ""
read -p "Making source build after pressing [Enter].."
echo 
python setup.py sdist
echo ""
read -p "Press [Enter] to install the pip package..."
pip install -U dist/bnlearn-0.1.0-py3-none-any.whl
echo ""
read -p ">twine upload dist/* TO UPLOAD TO PYPI..."
echo ""
read -p "Press [Enter] key to close window..."
