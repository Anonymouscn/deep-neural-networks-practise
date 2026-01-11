#!/bin/bash

conda env export > environment.yml # On conda environment
pip freeze > requirements.txt # On simple python environment