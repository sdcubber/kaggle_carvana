# -------------------------- #
# Run jobs on GPU machine
# -------------------------- #

# ---- helper functions ---- #

# Function to copy most recent log file to the dropbox folder
dropbox_log() {
	cd ../logs
	cp `ls -rtm1 | tail -1` /home/stijndc/Dropbox/Kaggle_carvana/logfiles/
	cd -
}

# Function to copy most recent set of test set predictions to dropbox
dropbox_predictions(){
	cd ../predictions/test
	cp `ls -rtm1 | tail -1` /home/stijndc/Dropbox/Kaggle_carvana/
	cd -
}

# Function to upload most recent log file to Google drive
gdrive_log(){
	cd ../logs
	~/gdrive-linux-x64 upload --parent 0B5Yph9ar5JREUHQzTXp2Tk9ZRDQ `ls -rtm1 | tail -1`
	cd -
}

# Function to upload most recent set of test predictions to Google drive
gdrive_predictions(){
	cd ../predictions/test
	~/gdrive-linux-x64 upload --parent 0B5Yph9ar5JREVzFmRmh0WTVOOGs `ls -rtm1 | tail -1`
	cd -
}

# ---------------------#

# ---- job script ---- #

# activate python environment
source activate pytorch

# Move to src directory
cd ../src

# run main script
# The message variable is not used for now (will be used to update spreadsheet later)
# Variable order: message im_size architecture epochs

MESSAGE='foo'

python main.py $MESSAGE 128 UNet_128_512_weights 1 -b 4 -db

#python evaluate.py 512 -b 4 

#gdrive_log
#gdrive_predictions

cd ../jobscripts

#sleep 30m # allow some time to upload files to dropbox
#shutdown now 
