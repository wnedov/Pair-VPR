#!/bin/bash
cd "$(dirname "$0")"
export OUTPUT_ROOT="runs/pairvpr_stagetwo_wildcross_alt"
sbatch eval_venman.sh
sbatch eval_karawatha.sh
sbatch eval_intra.sh