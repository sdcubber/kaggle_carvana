# -------------------------- #
# Run jobs on GPU machine
# -------------------------- #

# activate python environment
source activate pytorch

cd ../src

# Assign a name for each script. Output will be logged under that name
NAME='UNet_256'

# run python script
python main.py --arch ${NAME} 2>&1 | tee ../jobscripts/logs/${NAME}

cd ../jobscripts

