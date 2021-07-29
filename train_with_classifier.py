from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms

import torch.distributed as dist 
import apex.parallel import DistributedDataParallel



import matplotlib.pyplot as plt
import time
import os
import copy
import argparse


import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import utils as vutils
from tqdm import tqdm
from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

#torch.backends.cudnn.benchmark = True

def train_classifier(model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False):
    """
    Train the model that will be used later with GAN generated images to estimate at different times
    the GAN's discrete distribution frequencies of the classes.
    """
    
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_classifier(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model = torchvision.models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model = torchvision.models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model = torchvision.models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model = torchvision.models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model = torchvision.models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = torchvision.models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size


def tune_classifier(args, classifier, input_size, data_dir, device, batch_size, num_epochs):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])            
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])            
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Send the model to GPU
    classifier = classifier.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    print("Params to learn:")
    params_to_update = []
    for name, param in classifier.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


    # Setup the loss fxn
    weight = None
    if len(args.clf_loss_classes_weights) > 0:
        weight = args.clf_loss_classes_weights
        for c in '([])': weight = weight.replace(c, '')
        weight = [float(w) for w in weight.split(',')]
        weight = torch.tensor(weight).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    # Train and evaluate
    classifier, hist = train_classifier(classifier, dataloaders_dict, criterion, optimizer, device,
                                num_epochs=num_epochs, is_inception=(args.classifier.lower()=="inception"))

    return classifier, hist

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        part = np.random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()


def update_discrete_effective_class_distribution(N, C, alpha, beta):
    # for every class we calculate the effective class frequency
    for i in range(N.shape[0]):
        N[i] = (1-alpha) * N[i] + beta * C[i]
    # Normalize distribution
    N = N / N.sum()
    return N

"""
We added a classifier in the loop to be used for the loss regularizer
The important thing we need to consider is that G(noise) will generate images of dimension args.im_size
but the classifier knows how to classify images of dimension clf_im_size.
We will need to use one more transform operation to do that.
IMPORTANT: some image classifiers can work with variable input size like vgg or resnet
"""
def train(args, classifier, clf_im_size, devices):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    #nlr = 0.0002 # original
    nlr = 0.00002 # diet
    nbeta1 = 0.5
    multi_gpu = len(devices) > 1
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(args)
    
    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])            
        ]
    trans = transforms.Compose(transform_list)
    
    dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    
    
    #from model_s import Generator, Discriminator
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size)
    netD.apply(weights_init)

    netG.to(devices[0])
    netD.to(devices[0])
    classifier.to(devices[0])

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(args.fixed_samples, nz).normal_(0, 1).to(devices[0])

    if multi_gpu:
        print('Using models with DataParallel')
        netG = nn.DataParallel(netG, device_ids=devices)
        netD = nn.DataParallel(netD, device_ids=devices)
        classifier = nn.DataParallel(classifier, device_ids=devices)

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    
    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt

    # some variables to handle classes stats in the loop
    cycle = 0
    Na = 1.     # constant to initialize N distribution at t=0
    alpha = 0.5 # how much of the past we want to keep     (this should be 0.5 for good results)
    beta = 1.   # how much of the present we want to keep  (this should be 1.0 for good results)
    cycle_steps = args.cycle_steps
    N_dist = torch.ones(args.num_classes, requires_grad=False) * Na
    N_dist /= N_dist.sum()
    N_dist = N_dist.to(devices[0])
    _lambda = args.lreg_lambda

    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image = next(dataloader)
        #real_image = real_image.cuda(non_blocking=True)
        real_image = real_image.to(devices[0], non_blocking=True)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(devices[0])

        fake_images = netG(noise)
        # len of fake_images is 2
        # fake_images[0] should be a tensor of size [batch_size, 3, im_size, im_size]
        # fake_images[1] should be a tensor of size [batch_size, 3, im_size/2, im_size/2]

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

        if iteration % cycle_steps == 0:
            # we cycle increment
            cycle += 1

            if iteration != current_iteration:
                # we use previous N discrete class distribution and C to calculate a new N distribution
                print('update discrete effecting class distribution')
                N_dist = update_discrete_effective_class_distribution(N_dist, C, alpha, beta)
                print(f'N_dist: {N_dist}')
                if args.dynamic_lambda.lower() in ['true', '1']:
                    print('dynamic lambda')
                    # lambda = 1 / sum(e^N)
                    _lambda = 1. / torch.sum(torch.exp(N_dist))
            else:
                # we will use first defined N
                pass                
            print(f'new cycle t={cycle}')
            # reset counter of classes
            #C = torch.zeros(args.num_classes, device=devices[0])
            # to avoid zero division
            C = torch.ones(args.num_classes, device=devices[0]) / 100

        ## 2. train Discriminator
        netD.zero_grad()
        #classifier.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label="real")
        train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        optimizerD.step()
        
        ## 3. train Generator
        netG.zero_grad()
        pred_g = netD(fake_images, "fake")        
        # Lreg calculation
        pred_classes_softmax = torch.exp(classifier(fake_images[0]))
        rho = pred_classes_softmax.mean(0)
        
        if args.diet.lower() in ['y', 'yes', '1']:
            for _ in range(args.num_classes-1):
                rho[torch.argmax(rho)] = rho[torch.argmax(rho)]*-0.0001
            L_reg = -(rho / N_dist).sum()
        else:
            L_reg = ((rho * torch.log(rho)) / N_dist).sum()        

        err_g = -pred_g.mean() + (_lambda / args.num_classes) * L_reg

        err_g.backward()
        optimizerG.step()

        # track of class statistics for later usage with class effective frequency distribution
        batch_labels = torch.zeros(pred_classes_softmax.shape)
        for j, idx in enumerate(pred_classes_softmax.argmax(1)):
            batch_labels[j, idx] = 1
    
        batch_labels = batch_labels.to(devices[0])
        # update class counter
        C += batch_labels.sum(0)

        fmt_counts = [int(c) for c in C.tolist()]
        print('t:', cycle, 'class counter:', fmt_counts, 'N_dist:', N_dist.tolist(), f'_lambda: {_lambda:.4f}', f'L_reg: {L_reg:.4f}', f'err_g: {err_g:.4f}') 

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -err_g.item()))

        if iteration % (save_interval*10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder+'/%d.jpg'%iteration, nrow=8)
                vutils.save_image( torch.cat([
                        F.interpolate(real_image, 128), 
                        rec_img_all, rec_img_small,
                        rec_img_part]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )
            load_params(netG, backup_para)

        if iteration % (save_interval*50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Region GAN with help of a classifier')

    # I'm sure resnet and vgg will work with different input sizes but I'm not sure about the rest
    classifiers = ['squeezenet', 'resnet', 'alexnet', 'vgg', 'densenet', 'inception']

    parser.add_argument('--path', type=str, required=True, help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--num_classes', type=int, required=True, help='number of samples we expect to find in image folders')
    parser.add_argument('--classifier', type=str, default='resnet', choices=classifiers, help='classifier base to fine tune')
    parser.add_argument('--epochs', type=int, default=15, help='epochs to finetune the classifier')
    parser.add_argument('--lreg_lambda', type=float, default=1. , help='Used in loss as: lambda X Lreg')
    parser.add_argument('--dynamic_lambda', type=str, default='False', help='Lambda will be calculated automatically after each cycle when True')
    parser.add_argument('--feature_extract', type=str, default='True', help='Block classifier original weights before training')
    parser.add_argument('--cuda', type=str, default='0', help='indices of GPUs to be used')
    parser.add_argument('--name', type=str, default='gan_clf_test_1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations for GAN')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--cycle_steps', type=int, default=10, help='number of iterations between N_distribution updates')
    parser.add_argument('--im_size', type=int, default=256, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--fixed_samples', type=int, default=8, choices=[8, 16, 24, 32], help='Number of fixed samples to track generator behaviour')    
    parser.add_argument('--diet', type=str, default='False', help='Try to ignore most detected classes gradients')
    parser.add_argument('--clf_loss_classes_weights', type=str, default='',
                help='classifier cross_entropy_loss weights for classes (example [10, .1, .1] for ferrari, obama, pokemon)')

    args = parser.parse_args()
    print(args)

    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")

    # Initialize the model for this run
    feature_extract = args.feature_extract.lower() in ['true', '1']
    classifier, input_size = initialize_classifier(args.classifier.lower(), args.num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    # print(classifier)

    # Detect if we have a GPU available
    if ',' in args.cuda:
        cudas = args.cuda.split(',')
    elif ' ' in args.cuda:
         cudas = args.cuda.split(' ')
    else:
        cudas = [args.cuda]
    cudas = [int(c) for c in cudas]

    print(f"Devices we will try to use: {cudas}")
    if not torch.cuda.is_available():
        print('The script requires at least one GPU')
        sys.exit(1)
    print(f'Device to be used to fine tune the classifier: {cudas[0]}')

    # if we use the classifier original input size then later we will require to resize the output of the generator
    # to be able of classifying the images (generator output)
    # clf_im_size = input_size
    # but for the experiment we can try training the classifier with generator output size
    # (when using im_size = 256 will be not so much different of original classifier input size 224)
    # I don't remember all the classifiers that can handle dynamic image sizes
    # These classifiers are the ones that make use of Average Pooling layer on first layers
    clf_im_size = args.im_size
    classifier, hist = tune_classifier(args, classifier=classifier, input_size=clf_im_size, data_dir=args.path,
                                            device=cudas[0], batch_size=args.batch_size, num_epochs=args.epochs)
    classifier.eval()

    train(args=args, classifier=classifier, clf_im_size=clf_im_size, devices=cudas)
