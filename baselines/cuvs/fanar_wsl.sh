#!/bin/bash
export FANAR_ROOT="${FANAR_ROOT:-$(pwd)}"   # repo root (run from it)
# FanarVSE under WSL2 (same torch 2.6 / Triton 3.2 as native Windows) for the OS-overhead comparison
source ~/fanar_t/bin/activate; export FANAR=$FANAR_ROOT CC=~/bin/zigcc
cd $FANAR_ROOT && python -W ignore -u src/rev_fanardb.py platform 2>&1 | tail -n 60
