#!/bin/bash

set -e

watch -c -n 1 "squeue -p gpu -o '%.20i %.20j %.10u %.4t %.11M   %R'"