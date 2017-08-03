# -------------------------- #
# Run jobs on GPU machine
# -------------------------- #

source activate pytorch

# Assign a name for each script. Output will be logged under that name
NAME='UNet_256'

cd ../src
echo 'Running: '$NAME
python UNet.py $NAME 25 256 -b 16 2>&1 | tee ../jobscripts/logs/${NAME}
echo 'Done.'

cd ../jobscripts/logs
