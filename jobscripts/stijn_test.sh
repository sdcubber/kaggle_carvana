# -------------------------- #
# Run jobs on GPU machine
# -------------------------- #

# activate python environment
source activate pytorch

# Assign a name for each script, output and error files will be stored with that name
NAME='test'


cd ../src
echo 'Running: '$NAME
python UNet.py $NAME 10 -b 32  > ../jobscripts/logs/${NAME} 2> ../jobscripts/logs/${NAME}_err
echo 'Done.'

cd ../jobscripts/logs

