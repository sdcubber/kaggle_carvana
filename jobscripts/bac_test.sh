# -------------------------- #
# Run jobs on GPU machine
# -------------------------- #

# activate python environment
source activate pytorch

cd ../src

# Assign a name for each script. Output will be logged under that name
NAME='UNet_256_512'

# run python script
python main.py $NAME 512 UNet_128 1 -b 4 -db

# Copy log file to dropbox
# TODO

cd ../jobscripts

