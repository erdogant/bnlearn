function mkmodule() {
    mkdir $1
    touch $1/__init__.py
}

function pipa() {
    package_name=$1
    requirements_file=$2
    if [[ -z $requirements_file ]]
    then
        requirements_file='./requirements.txt'
    fi
    package_string=`pip freeze | grep -i $package_name`
    current_requirements=`cat $requirements_file`
    echo "$current_requirements\n$package_string" | LANG=C sort | uniq > $requirements_file
}

function pipia() {
    package_name=$1
    requirements_file=$2
    if [[ -z $requirements_file ]]
    then
        requirements_file='./requirements.txt'
    fi
    pip install $package_name
    pipa $package_name $requirements_file
}