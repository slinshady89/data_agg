import argparse


def argparser():
    # command line argments
    parser = argparse.ArgumentParser(description = 'Segmentation U-Net inference. '
                                                   'Predicting drivable paths in input images and highlight obstacles')
    parser.add_argument("--base_dir",
                        default = '/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/',
                        help = "base dir where labels are found")
    parser.add_argument("--sequence",
                        default = '08/',
                        help = "Sequence in case of KITTI data")
    parser.add_argument("--images",
                        default = '/image_2/',
                        help = "loading dir of true image")
    parser.add_argument("--gt_labels",
                        default = '/labels_gt/',
                        help = "loading dir of created labels by lidar and tf")
    parser.add_argument("--inf_labels",
                        # default = '20190729_164658/',
                        default = '20190726_130705/',
                        help = "loading dir of labely created by inference")

    args = parser.parse_args()

    return args
