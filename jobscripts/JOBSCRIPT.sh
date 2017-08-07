# -------------------------- #
# Run jobs on GPU machine
# -------------------------- #

# activate python environment
source activate pytorch

# Move to src directory
cd ../src

# run main script
# The message variable is not used for now (will be used to update spreadsheet later)
# Variable order: message im_size architecture epochs

MESSAGE='foo'
python main.py $MESSAGE 128 UNet_128 30 -b 32 

source deactivate

# Copy log file to dropbox
cd ../logs
cp `ls -rtm1 | tail -1` /home/stijndc/Dropbox/Kaggle_carvana/logfiles/
cd ../jobscripts

