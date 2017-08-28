# Model utility functions such as loss functions, CNN building blocks etc.
from data.config import *
from processing.processing_utils import AverageMeter
import processing.processing_utils as pu
import torchvision.transforms as transforms
from torch.nn.modules.loss import _Loss, _WeightedLoss
from PIL import Image
import processing.augmentation as pa

# --- Custom loss functions --- #
class DiceLoss(_Loss):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        return 1 - torch.mean(2 * torch.sum(input * target, 1) \
                                / (torch.sum(input, 1) + torch.sum(target,1))) \
               + F.binary_cross_entropy(input, target)


class BCELoss2D(_WeightedLoss):
    def __init__(self):
        super(BCELoss2D, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, input, target):
        return self.bce(input.view(-1), target.view(-1))


def predict(model, test_loader, args, log=None):
    """Make rle-encoded predictions on a set of data. Returns image ids and encoded predictions."""

    # switch to evaluate mode
    model.eval()

    # define output
    test_idx = []
    rle_encoded_predictions = []
    output_list=[]

    # number of iterations before print outputs
    print_iter = np.ceil(len(test_loader.dataset)/(10 * test_loader.batch_size))
    num_test = 0

    for batch_idx, (input, target, weight, id) in enumerate(test_loader):
        # forward + backward + optimize
        input_var = Variable(input.cuda() if GPU_AVAIL else input, volatile=True)

        # compute output
        output = model(input_var)

        # Don't forget to squeeze!!
        img_list = [np.squeeze(output.data[b].cpu().numpy()) for b in range(input.size(0))]

        # resize the test images
        img_list = [(pa.resize_cv2(item, O_HEIGH, O_WIDTH) > THRED).astype(np.uint8) for item in img_list]        # convert to {0,1} prediction

        # rle encode the predictions
        rle_encoded_predictions.extend([pu.rle_encode(np.array(item)) for item in img_list])
        test_idx.extend(id)

        # write to the log file
        num_test += input.size(0)
        if (batch_idx + 1) % print_iter == 0:
            log.write('Predicting {:>3.0f}%\n'.format(100*num_test/len(test_loader.dataset)))

        if args.store_probabilities:
            output_list.append(output.data.cpu().numpy()) # don't collect the list in gpu memory!

    return test_idx, rle_encoded_predictions, output_list


def train(train_loader, valid_loader, model, criterion, optimizer, args, log=None):
    """ Training the model """
    # load the last run
    best_dice, best_loss = load_checkpoint(args, model, optimizer, log)

    plateau_counter = 0
    lr_patience = 10 # patience for lr scheduling
    early_stopping_patience = 6 # patience for early stopping

    for epoch in range(args.start_epoch, args.epochs):

        if plateau_counter > early_stopping_patience:
            log.write('Early stopping: patience reached.')
            break
        # training the model
        run_epoch(train_loader, model, criterion, optimizer, epoch, args.epochs, args.n_acc,log)

        # validate the valid data
        valid_dice, valid_loss = evaluate(model, valid_loader, criterion)
        log.write("valid_loss={:.5f}, valid_dice={:.5f} \n".format(valid_loss, valid_dice))

        # remember best dice and save checkpoint
        is_best = (valid_dice > best_dice) or \
                  (valid_dice == best_dice and valid_loss < best_loss)

        best_dice = max(valid_dice, best_dice)
        best_loss = min(valid_loss, best_loss)

        # Save only the best state. Update each time the model improves
        bestpoint_file = os.path.join(OUTPUT_WEIGHT_PATH, '{}_best_architecture.pth.tar'
                                   .format(model.modelName))
        best_weight_file = os.path.join(OUTPUT_WEIGHT_PATH, '{}_best_weights.torch'
                                    .format(model.modelName))

        if not is_best:
            plateau_counter+=1
        else:
            plateau_counter=0

        if plateau_counter > lr_patience:
            log.write('Validation loss reached plateau. Reducing learning rate...')
            optimizer = adjust_lr_on_plateau(optimizer)

        if is_best: # Save only if it's the best model
            log.write('Saving best model architecture...\n')
            torch.save({
                'epoch': epoch + 1,
                'arch': model.modelName,
                'state_dict': model.state_dict(),
                'best_dice': best_dice,
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, bestpoint_file)

        # saving the best weights
        if is_best:
            log.write("Saving the best weights...\n")
            torch.save(model, best_weight_file)

        log.write("----------------------------------------------------------\n")

    # load the best model
    model = torch.load(os.path.join(OUTPUT_WEIGHT_PATH, '{}_best_weights.torch'.format(model.modelName)))

    # Put best validation loss in the names of best architecture and best weight files
    shutil.move(bestpoint_file, os.path.join(OUTPUT_WEIGHT_PATH, '{}_best_architecture_{}.pth.tar'
                               .format(model.modelName, best_loss)))

    shutil.move(best_weight_file, os.path.join(OUTPUT_WEIGHT_PATH, '{}_best_weights_{}.torch'
                                .format(model.modelName, best_loss)))

    return best_dice, best_loss

def run_epoch(train_loader, model, criterion, optimizer, epoch, num_epochs, n_acc, log=None):
    """Run one epoch of training."""
    # switch to train mode
    model.train()

    # define loss and dice recorder
    losses = AverageMeter()
    dices = AverageMeter()

    # number of iterations before print outputs
    print_iter = np.ceil(len(train_loader.dataset) / (10 * train_loader.batch_size))

    for batch_idx, (input, target, weight, id) in enumerate(train_loader):

        input_var = Variable(input.cuda() if GPU_AVAIL else input)
        target_var = Variable(target.cuda() if GPU_AVAIL else target)

        # compute output
        output = model(input_var)
        if train_loader.dataset.weighted:
            criterion.bce.weight = weight.view(-1).cuda() if GPU_AVAIL else weight.view(-1)

        #loss = criterion(output, target_var)
        # measure dice and record loss
        #score = get_dice_score(output.data.cpu().numpy(), target.cpu().numpy(), THRED)
        #dices.update(score, input.size(0))
        #losses.update(loss.data[0], input.size(0))

        # Zero gradients, compute gradients and do SGD step
        #loss.backward()

        optimizer.zero_grad()

        mini_weight = 0
        mini_dice = 0

        for i in range(input.size(0)):
            loss = torch.addcmul(criterion(output[i], target_var[i]), 1/input.size(0))
            loss.backward()
            mini_weight += loss.data[0]
            mini_dice += get_dice_score(output[i].data.cpu().numpy(), target[i].cpu().numpy(), THRED)

        optimizer.step()
        losses.update(mini_weight, input.size(0))
        dices.update(mini_dice, input.size(0))

        """
        # Gradient accumulation: do update step only once each n_acc epochs
        if batch_idx % n_acc == 0:
            optimizer.step()
            optimizer.zero_grad()
        """

        if (batch_idx + 1) % print_iter == 0:
            log.write('Epoch [{:>2}/{:>2}] {:>3.0f}%'
                      '\ttrain_loss={:.6f}, train_dice={:.6f}\n'.format(
                epoch + 1, num_epochs,
                100 * losses.count / len(train_loader.dataset),
                losses.avg, dices.avg))


def evaluate(model, data_loader, criterion):
    """ Evaluate model on labeled data. Used for evaluating on validation data. """

    # switch to evaluate mode
    model.eval()

    # define loss and dice recorder
    losses = AverageMeter()
    dices = AverageMeter()

    for batch_idx, (input, target, weight, id) in enumerate(data_loader):
        # forward + backward + optimize
        input_var = Variable(input.cuda() if GPU_AVAIL else input, volatile=True)
        target_var = Variable(target.cuda() if GPU_AVAIL else target, volatile=True)

        # compute output
        output = model(input_var)
        if data_loader.dataset.weighted:
            criterion.bce.weight = weight.view(-1).cuda() if GPU_AVAIL else weight.view(-1)
        loss = criterion(output, target_var)

        # measure dice and record loss
        score = get_dice_score(output.data.cpu().numpy(), target.cpu().numpy(), THRED)
        dices.update(score, input.size(0))
        losses.update(loss.data[0], input.size(0))

    return dices.avg, losses.avg

def load_checkpoint(args, model, optimizer, log=None):
    """
    Load saved parameters
    Return the best dice score and the best loss score
    """

    if args.resume and os.path.isfile(args.resume):
        log.write("=> loading checkpoint '{}'\n".f64ormat(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        args.arch = checkpoint['arch']
        best_dice = checkpoint['best_dice']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log.write("=> loaded checkpoint '{}' (epoch {})\n"
                  .format(args.resume, checkpoint['epoch']))
    else:
        log.write("=> no checkpoint found at '{}'\n".format(args.resume))
        best_dice = -float("inf")
        best_loss = float("inf")

    return best_dice, best_loss

def adjust_learning_rate(optimizer, epoch, init_lr=0.01, value=0.5):
    """Sets the learning rate to the initial LR decayed by value every 20 epochs"""
    lr = init_lr * (value ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_lr_on_plateau(optimizer):
    """Decrease learning rate by factor 10 if validation loss reaches a plateau"""
    for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/10
    return optimizer

def dice(im1, im2, empty_score=1.0):
    """ Return dice accuracy """
    im1 = im1.astype(np.bool)
    im2 = im2.astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum


def get_dice_score(avg_masks, train_masks, thr=0.5):
    """Return dice score"""
    score = 0.0
    predict_masks = avg_masks > thr

    for i in range(train_masks.shape[0]):
        score += dice(train_masks[i], predict_masks[i])

    return score / train_masks.shape[0]
