#!/bin/bash

path=$(pwd)
rm -rf "$path"/docs/build/*

latexmk -xelatex "$path"/docs/MLP.tex -outdir="$path"/docs/build
cp "$path"/docs/build/MLP.pdf "$path"/docs/report/MLP.pdf
rm -rf "$path"/docs/build/*