# Evaluate pretrained model

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

def main():
    prs = argparse.ArgumentParser(description='Load and evaluate trained model.')
    prs.add_argument('im_size', default=256, type=int, help='image size (default: 256)')
    prs.add_argument('-rot', '--rotation', default=None, type=int, help='Type of car rotation. Default None returns all rotations.')
    prs.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='Mini-batch size (default: 16)')
    prs.add_argument('-j', '--workers', default=3, type=int, metavar='N', help='Number of data loading workers')
    prs.add_argument('-sp', '--store_probabilities', action='store_true', help='Store predicted probabilities')
    prs.add_argument('-db', '--debug', action='store_true', help='Debug mode.')


    evaluate(prs)

def evaluate(parser):
    args = parser.parse_args()
    timestamp = datetime.now()

    file = 'log_evaluate'
    log = pu.Logger()
    log.open(os.path.join(OUTPUT_LOG_PATH, file), mode='w')

    log.write(str(args) + "\n")

    filename = 'UNet_128_1024_best_weights'
    model = torch.load('../models/{}.torch'.format(filename))

    if GPU_AVAIL:
        model = model.cuda()
        log.write('Using GPU...\n')

    # Type of car rotations
    rot_id = [args.rotation] if args.rotation else range(1, 17)

    # Data augmentation
    common_trans = None
    input_trans = transforms.Compose([
        transforms.Lambda(lambda x: pa.resize_cv2(x, args.im_size, args.im_size)),
    ])
    mask_trans = transforms.Compose([
        transforms.Lambda(lambda x: pa.resize_cv2(x, args.im_size, args.im_size)),
    ])

    # Dataset loaders
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
    output_file_train = os.path.join(OUTPUT_SUB_PATH, 'train', 'evaluate_TRAIN_{}.gz'
                                     .format(timestamp.strftime('%H_%M_%d_%m_%Y_{}'.format(filename))))
    output_file = os.path.join(OUTPUT_SUB_PATH, 'test', 'evaluate_{}_.gz'
                               .format(timestamp.strftime('%H_%M_%d_%m_%Y_{}'.format(filename))))

    log.write('Writing encoded csv files for training data..\n')
    pu.make_prediction_file(output_file_train, TRAIN_MASKS_CSV, train_idx, rle_encoded_predictions_train)
    log.write('Writing encoded csv files for test data...\n')
    pu.make_prediction_file(output_file, SAMPLE_SUB_CSV, test_idx, rle_encoded_predictions)
    log.write('Done!')
    
if __name__ == '__main__':
    # random.seed(123456789) # Fix seed
    sys.exit(main())
