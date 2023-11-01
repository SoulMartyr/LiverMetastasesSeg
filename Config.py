import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# data of dataset
parser.add_argument(
    '--data_dir', default="./data/resection_V", help='dataset root path')
parser.add_argument('--image_dir', default="images", help='image directory')
parser.add_argument('--mask_dir', default='liver_masks', help='mask directory')
parser.add_argument(
    '--index_path', default="./data/index_V.csv", help='index file path')
parser.add_argument('--fold',  nargs='+',  required=True, type=int,
                    help="Split number (0 to 4)")
parser.add_argument('--num_classes', type=int, default=1,
                    help='classes num to segment')
parser.add_argument(
    '--norm', choices=['zscore', 'minmax'], default='zscore',  help='using zscore or minmax to normalize')
parser.add_argument(
    '--img_d', type=int, default=-1,  help='resize image depth, -1 means no change')
parser.add_argument(
    '--img_h', type=int, default=224,  help='resize image height, -1 means no change')
parser.add_argument(
    '--img_w', type=int, default=224,  help='resize image width, -1 means no change')
parser.add_argument("--roi_z", default=32, type=int,
                    help="roi size in x direction")
parser.add_argument("--roi_y", default=224, type=int,
                    help="roi size in y direction")
parser.add_argument("--roi_x", default=224, type=int,
                    help="roi size in z direction")
parser.add_argument(
    '--keyframe', action='store_true', help='using key frame to train')
parser.add_argument(
    '--softmax', action='store_true', help='using softmax to process pred and mask ')
parser.add_argument(
    '--flip', action='store_true', help='flip data')

# train
parser.add_argument('--gpu', action='store_true',
                    help='whether use gpu')
parser.add_argument('--devices', type=int,  nargs='+',  required=True,
                    default=[0], help='which gpu to use')
parser.add_argument('--model', default="UNet3D", help='model')
parser.add_argument('--in_channels', type=int, default=1,
                    help='input channel')
parser.add_argument('--batch_size', type=int,
                    default=4, help='train batch size')
parser.add_argument('--epoch_num', type=int, default=600,
                    metavar='N', help='number of epochs to train (default: 501)')
parser.add_argument('--lr', type=float, default=0.001,
                    metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--optim', choices=['adam', 'sgd', 'adamw'],
                    default='adam',  help='optimizer (default: adam)')
parser.add_argument('--weight_decay', default=1e-5, type=float,
                    metavar='WD', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--warm_restart', action='store_true',
                    help='use scheduler warm restarts with period of 30')
parser.add_argument('--loss', type=str, nargs='+',
                    default=["bce", "dice"], help='loss type')
parser.add_argument('--class_weight', type=float, nargs='*',
                    default=[], help='weight for each class')
parser.add_argument('--background', action='store_true',
                    help='whether include background')
parser.add_argument('--loss_weight', type=float, nargs='*',
                    default=[], help='weight for loss')
parser.add_argument('--num_workers', default=8, type=int,
                    help='dataloader num workers (default: 8)')
parser.add_argument('--valid_epoch', type=int, default=5,
                    help='validate epoch interval(default: 5)')
parser.add_argument('--log_iter', type=int, default=10,
                    help='log information iteration interval(default: 10)')
parser.add_argument('--log_dir', default='./logs', help="log file path ")
parser.add_argument('--log_folder', type=str,
                    default="default", help='log folder name')
parser.add_argument('--lock', action='store_true',
                    help='lock log dir to avoid accidental deletion')
parser.add_argument('--use_ckpt', action='store_true',
                    help='use checkpoint to initial weight')
parser.add_argument('--ckpt_path', type=str,
                    default='./checkpoints/default/model.pth', help="pretrain checkpoint path")
parser.add_argument('--overlap', default=0.5, type=float,
                    help='valid window slide overlap (default: 0.5)')

# threshold
parser.add_argument('--thres', type=float, nargs='*',  default=[],
                    help='threshold for judging class')

# metrics
parser.add_argument('--metrics',  type=str,  nargs='+',
                    default=["dice_per_case", "dice_global", "voe_per_case", "voe_global", "rvd_per_case", "rvd_global", "asd"], help='metrics type')

# predict
parser.add_argument('--pred_dir', type=str, default='./predicts',
                    help='predict mask directory')

# addititon information
parser.add_argument('--add', type=str, default="", help='addition information')
args = parser.parse_args()
