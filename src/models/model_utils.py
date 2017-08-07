# Model utility functions such as loss functions, CNN building blocks etc.
from data.config import *
from processing.processing_utils import AverageMeter
import processing.processing_utils as pu
import torchvision.transforms as transforms
from torch.nn.modules.loss import _Loss


# --- Custom loss functions --- #
class DiceLoss(_Loss):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        return 1 - torch.mean(2 * torch.sum(input * target, 1) \
                                / (torch.sum(input, 1) + torch.sum(target,1)))


def predict(model, test_loader, log=None):
    """ Return the ids and their encoded predictions"""

    # switch to evaluate mode
    model.eval()

    # define output
    test_idx = []
    rle_encoded_predictions = []

    # number of iterations before print outputs
    print_iter = len(test_loader.dataset) // (10 * test_loader.batch_size)
    num_test = 0

    for batch_idx, (input, target, id) in enumerate(test_loader):
        # forward + backward + optimize
        input_var = Variable(input.cuda() if GPU_AVAIL else input, volatile=True)

        # compute output
        output = model(input_var)

        # Go from pytorch tensor to list of PIL images, which can be rescaled and interpolated
        PIL_list = [transforms.ToPILImage()(output.data[b].cpu()) for b in range(input.size(0))]

        # Rescale them to np matrices with the correct size
        np_list = [pu.upscale_test_img(img) for img in PIL_list]

        # rle encode the predictions
        rle_encoded_predictions.append([pu.rle(im >= 0.5) for im in np_list])
        test_idx.append(id)

        # write to the log file
        num_test += input.size(0)
        if (batch_idx + 1) % print_iter == 0:
            log.write('Predicting {:>3.0f}%\n'.format(100*num_test/len(test_loader.dataset)))

    return test_idx, rle_encoded_predictions


def train(train_loader, valid_loader, model, criterion, optimizer, args, log=None):
    """ Training the model """
    # load the last run
    best_dice, best_loss = load_checkpoint(args, model, optimizer, log)

    for epoch in range(args.start_epoch, args.epochs):
        # update learning rate
        adjust_learning_rate(optimizer, epoch, args.lr)

        # training the model
        run_epoch(train_loader, model, criterion, optimizer, epoch, args.epochs, log)

        # validate the valid data
        valid_dice, valid_loss = evaluate(model, valid_loader, criterion)
        log.write("valid_loss={:.5f}, valid_dice={:.5f} \n".format(valid_loss, valid_dice))

        # Save only the best state. Update each time the model improves
        checkpoint_file = os.path.join(OUTPUT_LOG_PATH, 'checkpoint_{}_{}_{}.pth.tar'
                                    .format(model.modelName, epoch, valid_dice))
        bestpoint_file = os.path.join(OUTPUT_LOG_PATH, 'modelbest_{}.pth.tar'
                                   .format(model.modelName))

        # remember best dice and save checkpoint
        is_best = (valid_dice > best_dice) or \
                  (valid_dice == best_dice and valid_loss < best_loss)

        best_dice = max(valid_dice, best_dice)
        best_loss = min(valid_loss, best_loss)

        if is_best: # Save only if it's the best model
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': model.modelName,
                'state_dict': model.state_dict(),
                'best_dice': best_dice,
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint_file, bestpoint_file)

        # saving the best weights
        if is_best:
            log.write("Saving the best weights...\n")
            torch.save(model, os.path.join(OUTPUT_WEIGHT_PATH, 'best_{}.torch'.format(model.modelName)))

        log.write("----------------------------------------------------------\n")

    # load the best model
    model = torch.load(os.path.join(OUTPUT_WEIGHT_PATH, 'best_{}.torch'.format(model.modelName)))

    return best_dice, best_loss

def run_epoch(train_loader, model, criterion, optimizer, epoch, num_epochs, log=None):
    """Run one epoc of training."""
    # switch to train mode
    model.train()

    # define loss and dice recorder
    losses = AverageMeter()
    dices = AverageMeter()

    # number of iterations before print outputs
    print_iter = len(train_loader.dataset) // (10 * train_loader.batch_size)

    for batch_idx, (input, target, id) in enumerate(train_loader):

        input_var = Variable(input.cuda() if GPU_AVAIL else input)
        target_var = Variable(target.cuda() if GPU_AVAIL else target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure dice and record loss
        score = get_dice_score(output.data.cpu().numpy(), target.cpu().numpy())
        dices.update(score, input.size(0))
        losses.update(loss.data[0], input.size(0))

        # Zero gradients, compute gradients and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    for batch_idx, (input, target, id) in enumerate(data_loader):
        # forward + backward + optimize
        input_var = Variable(input.cuda() if GPU_AVAIL else input, volatile=True)
        target_var = Variable(target.cuda() if GPU_AVAIL else target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure dice and record loss
        score = get_dice_score(output.data.cpu().numpy(), target.cpu().numpy())
        dices.update(score, input.size(0))
        losses.update(loss.data[0], input.size(0))

    return dices.avg, losses.avg


def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def load_checkpoint(args, model, optimizer, log=None):
    """
    Load saved parameters
    Return the best dice score and the best loss score
    """

    if args.resume and os.path.isfile(args.resume):
        log.write("=> loading checkpoint '{}'\n".format(args.resume))
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
    predict_masks = avg_masks >= thr

    for i in range(train_masks.shape[0]):
        score += dice(train_masks[i], predict_masks[i])

    return score / train_masks.shape[0]
