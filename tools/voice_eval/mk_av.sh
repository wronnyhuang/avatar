# mk_av.sh <rec.mkv> <out.mp4> [skew]: stage crop + audio shifted by the capture skew (from the flash calibration)
ffmpeg -v error -y -i "$1" -itsoffset ${3:-3.345} -i "$1" -map 0:v:0 -map 1:a:0 -vf crop=740:416:13:127 -c:v libx264 -preset ultrafast -crf 16 -c:a aac -b:a 128k -shortest "$2"
