# assuming this script is in the same directory as Videos
# export PATH=$PATH:"./SAIS/libraries/FFmpeg/bin" # (not sure if this is necessary - my guess is yes if stand-alone project)

# f is the filename flag
while getopts f: flag
do
    case "${flag}" in
        f) videoname=${OPTARG};;
    esac
done

mkdir "./SAIS/images";
# create image directory for each video (check on Linux)
# for FILE in "*.mp4";
#     do mkdir "./SAIS/images/${FILE%.*}"; 
# done
mkdir "./SAIS/images/$videoname" 

# Extract frames and save in video-specific directory (check on Linux)
# for FILE in "./SAIS/videos/*.mp4"; 
#     do ffmpeg -i $FILE "./SAIS/images/${FILE%.*}/frames_%8d.jpg"; 
# done
ffmpeg -i "./SAIS/videos/$videoname.mp4" "./SAIS/images/$videoname/frames_%8d.jpg"; 



