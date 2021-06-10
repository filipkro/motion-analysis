echo 'cloning repos'

git -C pose clone https://github.com/filipkro/mmpose.git
git -C pose/mmpose clone https://github.com/filipkro/mmdetection.git
git -C classification clone https://github.com/filipkro/tsc.git

echo 'repos cloned'
