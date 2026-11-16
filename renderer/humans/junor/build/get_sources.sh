#!/bin/bash
# junor: fetch + cache the large binary sources (kept out of git).  usage: get_sources.sh <T/asset-src>
set -e
D=${1:-asset-src}
mkdir -p "$D"; cd "$D"
[ -d ICT-FaceKit ] || git clone --depth 1 https://github.com/ICT-VGL/ICT-FaceKit.git
python3 -I "$(dirname "$(readlink -f "$0")")/ict_io.py" ICT-FaceKit ict.npz
echo "MPFB2 data (CC0): download https://github.com/makehumancommunity/mpfb2 and use src/mpfb/data as <mpfb_data_dir> (see LICENSES.md)"
