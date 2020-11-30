
data_name="lol" # (keypoints to use)
subject="36SLS1R_Oqus_2_14902.avi"
out_video="/home/filipkr/Documents/xjob/results-FL/results/outputFL.mp4"
in_video="/home/filipkr/Documents/xjob/results-FL/results/36SLS1R_Oqus_2_14902.avi"
out_data="/home/filipkr/Documents/xjob/results-FL/results/36SLS1R_3D.npy"


python run.py -d custom -k $data_name -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject $subject --viz-action custom --viz-camera 0 --viz-video $in_video --viz-output $out_video --viz-size 6 --viz-export $out_data

