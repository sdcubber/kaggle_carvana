from models.model_utils import *
from processing.processing_utils import *

from torch.utils.data import DataLoader
from data.data_utils import CarvanaDataset
from models.models import UNet128

parser = argparse.ArgumentParser(description='PyTorch UNet Training')

parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
                    help='model architecture ')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--img-size', default=256, type=int,
                    metavar='N', help='image size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--valid-size', default=0.1, type=float, metavar='M',
                    help='valid_size')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

def main():
    args = parser.parse_args()

    file = datetime.now().strftime('log_%H_%M_%d_%m_%Y_{}.log'.format(args.arch))
    log = Logger()
    log.open(os.path.join(OUTPUT_TEMP_PATH, file), mode='w')

    log.write(str(args) + "\n")

    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss()
    model = UNet128(args.arch)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if GPU_AVAIL:
        model = model.cuda()
        criterion = criterion.cuda()
        log.write("Using GPUs...")
    #--------------------------------TRAINING-----------------------------------------#
    #data augmentation
    input_trans = transforms.Compose([transforms.Scale(args.img_size),
                                      transforms.CenterCrop(args.img_size),
                                      transforms.ToTensor()])
    mask_trans = transforms.Compose([transforms.Scale(args.img_size),
                                     transforms.CenterCrop(args.img_size)])
    # types of rotations
    rot_id= range(1, 17)

    #split data set for training and valid
    train_ids, valid_ids = train_valid_split(TRAIN_MASKS_CSV, rotation_ids=rot_id,
                                             valid=args.valid_size)

    #preparing data flow for training the network
    dset_train = CarvanaDataset(im_dir=TRAIN_IMG_PATH,
                                ids_list=train_ids,
                                mask_dir=TRAIN_MASKS_PATH,
                                input_transforms=input_trans,
                                mask_transforms=mask_trans,
                                rotation_ids=rot_id,
                                debug=True)
    dset_valid = CarvanaDataset(im_dir=TRAIN_IMG_PATH,
                                ids_list=valid_ids,
                                mask_dir=TRAIN_MASKS_PATH,
                                input_transforms=input_trans,
                                mask_transforms=mask_trans,
                                rotation_ids=rot_id,
                                debug=True)

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

    best_dice, best_loss = train(train_loader, valid_loader, model, criterion, optimizer, args, log)
    #---------------------------------------------------------------------------------#

    #---------------------------------TESTING-----------------------------------------#
    dset_test = CarvanaDataset(im_dir=TEST_IMG_PATH,
                               input_transforms=input_trans,
                               rotation_ids=rot_id,
                               debug=True)
    test_loader = DataLoader(dset_test,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=GPU_AVAIL)

    test_idx, rle_encoded_predictions = predict(model, test_loader, log)
    output_file = join_path(OUTPUT_SUB_PATH, 'subm_{}_{:.5f}_{:.5f}.gz'
                            .format(model.modelName, best_dice, best_loss))
    make_prediction_file(output_file, test_idx, rle_encoded_predictions)
    # --------------------------------------------------------------------------------#

if __name__ == '__main__':

    random.seed(123456789)
    main()
