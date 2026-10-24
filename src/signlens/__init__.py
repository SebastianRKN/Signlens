import os

# Must be set before OpenCV is imported: Windows' Media Foundation backend otherwise spends up to a
# minute enabling hardware transforms on some webcams before the first frame.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
