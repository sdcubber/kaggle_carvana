# -------------------------- #
# Run jobs on GPU machine
# -------------------------- #

source activate pytorch

# Assign a name for each script. Output will be logged under that name
NAME='test'

cd ../src
echo 'Running: '$NAME
python UNet.py $NAME 1 128 -b 32 -db 2>&1 | tee ../jobscripts/logs/${NAME}

echo 'Gzipping submission file...'
gzip ../predictions/test/${NAME}.csv ../predictions/test/${NAME}.csv.gz

echo 'Done.'

cd ../jobscripts/logs
