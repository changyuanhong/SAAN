# -----------------------------------------------------------------------------------
#   * define parameter function
# -----------------------------------------------------------------------------------

import argparse


def str2bool(v):
    return v.lower() in ('true')


def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters


    parser.add_argument('--z_dim', type=int, default=128)

    parser.add_argument('--size', type=int, default=1064)

    parser.add_argument('--lambda_gp', type=float, default=5)

    # HP_a_1_4s

    parser.add_argument('--version', type=str, default='debug')

    parser.add_argument('--batch_size', type=int, default =10)

    parser.add_argument('--dataset', type=str, default='sq')

    parser.add_argument('--num_classes', type=int, default =1 )
    
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=5)

    # Training setting

    parser.add_argument('--iteration', type=int, default=1500,
                        help='epoch of updating GAN')

    parser.add_argument('--iteration_classifier', type=int, default = 6,
                        help='epoch of updating the C')

    parser.add_argument('--iteration_vae', type=int, default=800,
                        help='epoch of updating the vae')

    parser.add_argument('--lrG', type=float, default=0.0001)
    parser.add_argument('--lrD', type=float, default=0.0003)

    parser.add_argument('--lrC', type=float, default=0.0005)

    parser.add_argument('--lr_decay', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--test', type=str2bool, default=True)
    parser.add_argument('--classifier', type=str2bool, default=True)

    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    parser.add_argument('--cuda', default='true', action='store_true',
                        help='enables cuda')
    # Path
    parser.add_argument('--results', type=str,
                        default='F:/Works/Codes/GAN_revision_asset')

    parser.add_argument('--image_path', type=str, default='img')

    parser.add_argument('--model_save_path', type=str, default='models')

    parser.add_argument('--dataset_path', type=str, default='datasets')

    # Step size
    parser.add_argument('--log_step', type=int, default=5)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=float, default=1.0)

    return parser.parse_args()
