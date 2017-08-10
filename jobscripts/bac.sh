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

# ---------------------#

# ---- job script ---- #

# activate python environment
source activate pytorch

# Move to src directory
cd ../src

# run main script
# The message variable is not used for now (will be used to update spreadsheet later)
# Variable order: message im_size architecture epochs

MESSAGE='weigted_loss_test'

python main.py $MESSAGE 512 UNet_128_512_50epochs 30 -b 4 -we

dropbox_log
dropbox_predictions


cd ../jobscripts

sleep 30m # allow some time to upload files to dropbox
shutdown now 
