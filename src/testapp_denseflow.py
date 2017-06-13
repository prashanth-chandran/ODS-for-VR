from viewSynth import *
from SJPImage import *
import argparse
import matplotlib.pyplot as pyplt


def arg_setup():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--first", required=True, help="path to image file")
    # ap.add_argument("-s", "--second", required=True, help="path to calibration file")
    args = vars(ap.parse_args())
    return args



def test_optical_flow():
    """
    Function to test optical flow
    """
    args = arg_setup()
    image_file = args["first"]

    ic0 = SJPImageCollection()
    ic0.loadImagesFromYAML(image_file, 'frame0')

    of = OpticalFlowCalculator()
    im0 = ic0[7]
    im1 = ic0[4]
    flow_01 = of.calculateFlow(im0.getImage(), im1.getImage())
    warped_image = of.warpImageWithFlow(im1.getImage(), flow_01)
    flow_hsv = of.getFlowInHSV(flow_01)
    # Plot stuff
    pyplt.subplot(221), pyplt.imshow(im0.getImage()), pyplt.title('Frame 1'), pyplt.axis('off')
    pyplt.subplot(222), pyplt.imshow(im1.getImage()), pyplt.title('Frame 2'), pyplt.axis('off')
    pyplt.subplot(223), pyplt.imshow(warped_image), pyplt.title('Warped Image'), pyplt.axis('off')
    pyplt.subplot(224), pyplt.imshow(flow_hsv), pyplt.title('Flow magnitude'), pyplt.axis('off')
    pyplt.show()


def main():
    test_optical_flow()


if __name__ == '__main__':
    main()

