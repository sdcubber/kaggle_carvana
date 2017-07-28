kaggle_carvana
==============================

Carvana Image Masking Challenge

Project Organization
------------
- data: organize the data in the following way:
  - /data/raw/train/: training set images
  - /data/raw/test/: test set images
  - /data/raw/train_masks/: training set masks (.gif files)
  - /data/train_masks.csv
  - /data/sample_submission.csv
  - /data/metadata.csv
- src: put Python code here
- models: store model weights and architectures here
- predictions: prediction files for train and test data
- jobscripts: put jobscripts to run on GPU machine. See example jobscript.

***! Note that the folder /data/ is ignored by github. If you add additional data to this folder locally, you will have to do the same on the GPU machine. The same is true for /notebooks/.***


'Submit jobs' to the GPU
------------
See the example jobscript. Write a jobscript, push it to the repository and I will pull it and run it on the GPU.

Use relative paths everywhere!
------------
For instance, from the directory /src/, read in the training data from '../data/raw/train/'. Use relative paths everywhere, and the code will run on any machine.

git workflow (to push to master):
------------
1. ```git pull``` to pull most recent version of repository
2. Do work
3. ```git status``` to see which files have changed
4. ```git add ./filesthatyouhavechanged```
5. ```git commit -m message```
5. ```git push```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
