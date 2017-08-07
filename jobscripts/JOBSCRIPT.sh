# -------------------------- #
# Run jobs on GPU machine
# -------------------------- #

# activate python environment
source activate pytorch

# Move to src directory
cd ../src

# run main script
python main.py 512 UNet_128 5 -b 4 -db

# Copy log file to dropbox
cd ../logs
cp `ls -rtm1 | tail -1` /home/stijndc/Dropbox/Kaggle_carvana/logfiles/
cd ../jobscripts
