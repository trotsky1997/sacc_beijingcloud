#!/usr/bin/env bash
echo $(cd $(dirname ${BASH_SOURCE[0]}); pwd)
echo -e "\n" >> ~/.bashrc
echo "alias sacc='python $(cd $(dirname ${BASH_SOURCE[0]}); pwd)/sacc.py'" >> ~/.bashrc
source ~/.bashrc