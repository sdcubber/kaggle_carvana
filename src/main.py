from data.config import *
# custom modules
import time
import models.models as mo
import models.model_utils as mu
import processing.processing_utils as pu
import processing.augmentation as pa
from data.data_utils import CarvanaDataset

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import h5py

def run_experiment(parser):
    args = parser.parse_args()
    timestamp = datetime.now()
    file = timestamp.strftime('log_%H_%M_%d_%m_%Y_{}.log'.format(args.arch))
    log = pu.Logger()
    log.open(os.path.join(OUTPUT_LOG_PATH, file), mode='w')

    log.write(str(args) + "\n")

    # define loss function (criterion) and optimizer
    criterion = mu.BCELoss2D()
    model = mo.UNet128(args.arch)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if GPU_AVAIL:
        model = model.cuda()
        criterion = criterion.cuda()
        log.write("Using GPU...")

    # --- TRAINING --- #

    # Type of car rotations
    rot_id = [args.rotation] if args.rotation else range(1, 17)

    # Data augmentation
    common_trans = transforms.Compose([
        transforms.Lambda(lambda (x,y): pa.randomBrightness(x,y, p=0.75)),
        transforms.Lambda(lambda (x,y): pa.randomHue(x,y,p=0.25)),
        transforms.Lambda(lambda (x,y): pa.randomHorizontalFlip(x,y, p=0.5)),
        transforms.Lambda(lambda (x,y): pa.randomHorizontalShift(x,y, p=0.5)),
        transforms.Lambda(lambda (x,y): pa.randomVerticalShift(x,y, p=0.5))
    ])
    input_trans = transforms.Compose([
        transforms.Lambda(lambda x: pa.resize_cv2(x, args.im_size, args.im_size)),
    ])
    mask_trans = transforms.Compose([
        transforms.Lambda(lambda x: pa.resize_cv2(x, args.im_size, args.im_size)),
    ])

    # split data set for training and valid
    train_ids, valid_ids = pu.train_valid_split(TRAIN_MASKS_CSV, rotation_ids=rot_id,
                                                valid=args.valid_size)

    # preparing data flow for training the network
    dset_train = CarvanaDataset(im_dir=TRAIN_IMG_PATH,
                                ids_list=train_ids,
                                mask_dir=TRAIN_MASKS_PATH,
                                common_transforms=common_trans,
                                input_transforms=input_trans,
                                mask_transforms=mask_trans,
                                rotation_ids=rot_id,
                                weighted=args.weighted,
                                debug=args.debug)

    dset_valid = CarvanaDataset(im_dir=TRAIN_IMG_PATH,
                                ids_list=valid_ids,
                                mask_dir=TRAIN_MASKS_PATH,
                                common_transforms=common_trans,
                                input_transforms=input_trans,
                                mask_transforms=mask_trans,
                                rotation_ids=rot_id,
                                weighted=args.weighted,
                                debug=args.debug)

    train_loader = DataLoader(dset_train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=GPU_AVAIL)
    valid_loader = DataLoader(dset_valid,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=GPU_AVAIL)

    start_time = time.time()
    best_dice, best_loss = mu.train(train_loader, valid_loader, model, criterion, optimizer, args, log)
    elapsed_time = time.time() - start_time
    print('Elapsed time for training: {} minutes'.format(np.round(elapsed_time/60, 2)))
    print('Time per epoch: {} seconds'.format(elapsed_time/args.epochs))
    # ---------------------------------------------------------------------------------#

    # --- TESTING --- #
    dset_train_full = CarvanaDataset(im_dir=TRAIN_IMG_PATH,
                                     input_transforms=input_trans,
                                     rotation_ids=rot_id,
                                     debug=args.debug)
    train_full_loader = DataLoader(dset_train_full,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=GPU_AVAIL)

    dset_test = CarvanaDataset(im_dir=TEST_IMG_PATH,
                               input_transforms=input_trans,
                               rotation_ids=rot_id,
                               debug=args.debug)
    test_loader = DataLoader(dset_test,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=GPU_AVAIL)

    log.write('Predicting training data...\n')
    train_idx, rle_encoded_predictions_train, output_train = mu.predict(model, train_full_loader, args, log)
    log.write('Predicting test data...\n')
    test_idx, rle_encoded_predictions, output_test = mu.predict(model, test_loader, args, log)

    # Store rle encoded outputs
    output_file_train = os.path.join(OUTPUT_SUB_PATH, 'train', 'TRAIN_{}_{:.5f}_{:.5f}.gz'
                                     .format(timestamp.strftime('%H_%M_%d_%m_%Y_{}'.format(args.arch)), best_dice,
                                             best_loss))
    output_file = os.path.join(OUTPUT_SUB_PATH, 'test', '{}_{:.5f}_{:.5f}.gz'
                               .format(timestamp.strftime('%H_%M_%d_%m_%Y_{}'.format(args.arch)), best_dice, best_loss))

    if args.store_probabilities:
        log.write('Storing predicted probabilities...\n')
        # Store output probabilities
        # see https://stackoverflow.com/questions/20928136/input-and-output-numpy-arrays-to-h5py
        # and https://stackoverflow.com/questions/22400652/compress-numpy-arrays-efficiently
        # for reading in the .h5 files
        h5f = h5py.File('../models/probabilities_{}_{:.5f}_{:.5f}.h5'.format(timestamp.strftime('%H_%M_%d_%m_%Y_{}'.format(args.arch)), best_dice,
                best_loss), 'w')
        h5f.create_dataset('TRAIN', data=np.concatenate([output_train], axis=0), compression='gzip', compression_opts=9)
        h5f.create_dataset('TEST', data=np.concatenate([output_test], axis=0), compression='gzip', compression_opts=9)
        h5f.close()

    log.write('Writing encoded csv files for training data..\n')
    pu.make_prediction_file(output_file_train, TRAIN_MASKS_CSV, train_idx, rle_encoded_predictions_train)
    log.write('Writing encoded csv files for test data...\n')
    pu.make_prediction_file(output_file, SAMPLE_SUB_CSV, test_idx, rle_encoded_predictions)
    log.write('Done!')
    # --------------------------------------------------------------------------------#

def main():
    prs = argparse.ArgumentParser(description='Kaggle: Carvana car segmentation challenge')
    prs.add_argument('message', default=' ', type=str, help='Message to describe experiment in spreadsheet')
    prs.add_argument('im_size', default=256, type=int, help='image size (default: 256)')
    prs.add_argument('arch', default='UNet', help='Model architecture ')
    prs.add_argument('epochs', default=30, type=int, help='Number of total epochs to run')
    prs.add_argument('-j', '--workers', default=3, type=int, metavar='N', help='Number of data loading workers')
    prs.add_argument('-lr', '--lr', default=0.01, type=float, metavar='LR', help='Initial learning rate')
    prs.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='Mini-batch size (default: 16)')
    prs.add_argument('-rot', '--rotation', default=None, type=int, help='Type of car rotation. Default None returns all rotations.')
    prs.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Manual epoch number (useful on restarts)')
    prs.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    prs.add_argument('--valid_size', default=0.1, type=float, metavar='M', help='Validation set size')
    prs.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    prs.add_argument('--resume', default='', type=str, metavar='PATH', help='Path to latest checkpoint (default: none)')
    prs.add_argument('-db', '--debug', action='store_true', help='Debug mode.')
    prs.add_argument('-we', '--weighted', action='store_true', help='Use weighted loss.')
    prs.add_argument('-sp', '--store_probabilities', action='store_true', help='Store predicted probabilities')
    run_experiment(prs)


if __name__ == '__main__':
    # random.seed(123456789) # Fix seed
    sys.exit(main())
