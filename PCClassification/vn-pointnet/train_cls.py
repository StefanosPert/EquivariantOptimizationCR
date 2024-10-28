
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
from data import ModelNet40
import math
import shutil
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

#torch.manual_seed(42)
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 24]')
    parser.add_argument('--model_mode', default='pointnet_equi', help='equivariant or not [default: pointnet_equi]')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='eqpointnet', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--resume',type=int)
    parser.add_argument('--data_dir',type=str,default='modelnet')
    parser.add_argument('--rot', type=str, default='aligned', metavar='N',
                        choices=['aligned', 'z', 'so3'],
                        help='Rotation augmentation to input data')
    parser.add_argument('--pooling', type=str, default='mean', metavar='N',
                        choices=['mean', 'max'],
                        help='VNN only: pooling method.')
    return parser.parse_args()

def test(model, loader,mix_coef,equiv, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        
        trot = None
        if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(points.shape[0])*360, axis="Z", degrees=True)
        elif args.rot == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]))
        if trot is not None:
            points = trot.transform_points(points)
        
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        
        pred,_,_= classifier(points,equiv,mix_coef)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)


    train_loader = torch.utils.data.DataLoader(ModelNet40(partition='train', num_points=args.num_point,base_dir=args.data_dir), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(ModelNet40(partition='test', num_points=args.num_point,base_dir=args.data_dir), num_workers=8,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)

    '''MODEL LOADING'''
    num_class = 40
    MODEL = importlib.import_module(args.model_mode + '.' + args.model)
    shutil.copy('./models/%s/%s.py' % (args.model_mode, args.model), str(experiment_dir))
    shutil.copy('./models/%s/pointnet_util.py' % args.model_mode, str(experiment_dir))

    classifier = MODEL.get_model(args, num_class, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct=[]
    if args.resume>=0:
        checkpoint=torch.load( str(checkpoints_dir) + f'/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch=args.resume
    if args.resume<0:
        with torch.no_grad():
            for name, param in classifier.named_parameters():
                
                if ('dense_lin' in name) or ('dense_map_to_dir' in name):

                    param.zero_()
    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        if epoch<=200/2:
            mix_coef=2*epoch/200
        else:
            mix_coef=1-2*(epoch-200/2)/200
        mix_coef=max(mix_coef,0)
        print("Mix coef",mix_coef)

        for batch_id, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            
            points, target = data

            
            trot = None
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(points.shape[0])*360, axis="Z", degrees=True)
            elif args.rot == 'so3':
                trot = Rotate(R=random_rotations(points.shape[0]))
            if trot is not None:
                points = trot.transform_points(points)
            
            points = points.data.numpy()

            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])

            points = torch.Tensor(points)
            target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()

            pred, trans_feat,n = classifier(points,False,mix_coef)
            loss = criterion(pred, target.long(), trans_feat)+0.01*(torch.sqrt(n[0])+torch.sqrt(n[1]))
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), test_loader,mix_coef=mix_coef,equiv=True)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler':scheduler.state_dict()
                }
                torch.save(state, savepath)
            savepath = str(checkpoints_dir) + f'/model_{epoch}.pth'
            log_string('Saving at %s'% savepath)
            state = {
                    'epoch': epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler':scheduler.state_dict()
                }
            torch.save(state,savepath)
            instance_acc, class_acc = test(classifier.eval(), test_loader,mix_coef=mix_coef,equiv=False)
            log_string('Test NoProj Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            global_epoch += 1

    logger.info('End of training...')
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
