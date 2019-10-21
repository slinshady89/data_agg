import argparse


def argparser():
    # command line argments
    parser = argparse.ArgumentParser(description = 'DAgger algorithm for data training model you desire. '
                                                   'Predicting drivable paths in input images and highlight obstacles')
    parser.add_argument("--base_dir",
                        default = '/media/localadmin/Test/11Nils/kitti/dataset/sequences/Data/',
                        help = "base dir where labels are found")
    parser.add_argument("--img_dir",
                        default = 'images/',
                        help = "loading dir of true image")
    parser.add_argument("--label_dir",
                        default = 'labels/',
                        help = "loading dir of ground truth labels")
    parser.add_argument("--inf_dir",
                        default = 'inf/',
                        help = "save dir of predicted labels")
    parser.add_argument("--dag_dir",
                        default = 'dagger/',
                        help = "dagger top folder")
    parser.add_argument("--poses_dir",
                        default = 'poses/',
                        help = "dagger top folder")

    args = parser.parse_args()

    return args
