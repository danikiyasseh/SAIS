# run the bash script from the root directory --> bash ./SAIS/main.sh

# the user must define the target videoname
while getopts f: flag
do
    case "${flag}" in
        f) videoname=${OPTARG};;
    esac
done

# # convert video to frames and place in images directory (DONE)
bash ./SAIS/scripts/video_to_frames.sh -f $videoname

# # generate paths to frames and flows and save as csv files (DONE)
python ./SAIS/scripts/generate_paths.py -f $videoname -p ./SAIS/

# # generate flow maps (DONE)
python ./SAIS/scripts/extract_representations.py --arch vit_small --patch_size 16 --model_type ViT_SelfSupervised_ImageNet --batch_size_per_gpu 2 --data_path ./SAIS/ --data_list 'Custom' --save_type h5 --optical_flow

# extract representations of rgb images (DONE)
python -m torch.distributed.launch ./SAIS/scripts/extract_representations.py --arch vit_small --patch_size 16 --model_type ViT_SelfSupervised_ImageNet --batch_size_per_gpu 1024 --data_path ./SAIS/ --data_list Custom --save_type h5

# # extract representations of flow maps (DONE)
python -m torch.distributed.launch ./SAIS/scripts/extract_representations.py --arch vit_small --patch_size 16 --model_type ViT_SelfSupervised_ImageNet --batch_size_per_gpu 256 --data_path ./SAIS/ --data_list Custom --save_type h5 --optical_flow_to_reps

# perform inference (DONE)
python -m torch.distributed.launch ./SAIS/scripts/run_experiments.py -p ./SAIS/ -data Custom_Gestures -d Custom -m ViT -enc ViT_SelfSupervised_ImageNet -t Prototypes -mod RGB-Flow -dim 384 -bs 2 -lr 1e-1 -nc 2 -bc -sa -domains in_vs_out -ph Custom_inference -dt reps -e 1 -f 1 --inference

# process inference results to generate valid predictions (DONE)
python ./SAIS/scripts/process_inference_results.py -p ./SAIS/
