from yolo.centroidtracker import CentroidTracker
from yolo.trackableobject import TrackableObject
import argparse
import cv2 as cv
import numpy as np
import os
import dlib

def Argument_Parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str,
                    help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
                    help="# of skip frames between detections")
    ap.add_argument("-df", "--datapath", type=str, default="./data/test.csv",
                    help="path to the data collection file")
    args = vars(ap.parse_args())
    print(args)
    return (args)


# Initialize the parameters
confThreshold = 0.6  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')

parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Load names of classes
classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "darknet/cfg/yolov3.cfg";
modelWeights = "darknet/yolov3.weights";

# load our serialized model from disk
print("[INFO] loading model...")
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

# initialize the video writer
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalDown = 0
totalUp = 0


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    # cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
    # Draw a center of a bounding box
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    cv.line(frame, (0, frameHeight // 2 - 50), (frameWidth, frameHeight // 2 - 50), (0, 255, 255), 2)
    cv.circle(frame, (left + (right - left) // 2, top + (bottom - top) // 2), 3, (0, 0, 255), -1)

    counter = []
    if (top + (bottom - top) // 2 in range(frameHeight // 2 - 2, frameHeight // 2 + 2)):
        coun += 1
        # print(coun)

        counter.append(coun)

    label = 'Pedestrians: '.format(str(counter))
    cv.putText(frame, label, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, detections):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    rects = []
    CLASSES= [ "person","bicycle","car","motorbike","aeroplane","toothbrush"]
# bus
# train
# truck
# boat
# traffic light
# fire hydrant
# stop sign
# parking meter
# bench
# bird
# cat
# dog
# horse
# sheep
# cow
# elephant
# bear
# zebra
# giraffe
# backpack
# umbrella
# handbag
# tie
# suitcase
# frisbee
# skis
# snowboard
# sports ball
# kite
# baseball bat
# baseball glove
# skateboard
# surfboard
# tennis racket
# bottle
# wine glass
# cup
# fork
# knife
# spoon
# bowl
# banana
# apple
# sandwich
# orange
# broccoli
# carrot
# hot dog
# pizza
# donut
# cake
# chair
# sofa
# pottedplant
# bed
# diningtable
# toilet
# tvmonitor
# laptop
# mouse
# remote
# keyboard
# cell phone
# microwave
# oven
# toaster
# sink
# refrigerator
# book
# clock
# vase
# scissors
# teddy bear
# hair drier

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated
        # with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by requiring a minimum
        # confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # detections list
            idx = int(detections[0, 0, i, 1])

            # if the class label is not a person, ignore it
            if CLASSES[idx] != "person":
                continue

            # compute the (x, y)-coordinates of the bounding box
            # for the object
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            # construct a dlib rectangle object from the bounding
            # box coordinates and then start the dlib correlation
            # tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(rgb, rect)

            # add the tracker to our list of trackers so we can
            # utilize it during skip frames
            trackers.append(tracker)

    for tracker in trackers:
        pos = tracker.get_position()
        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())
    rects.append((startX, startY, endX, endY))
    objects = ct.update(rects)
    counting(objects)

        # drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


def counting(objects):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    global totalDown
    global totalUp

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object

                if direction < 0 and centroid[1] in range(frameHeight // 2 - 30, frameHeight // 2 + 30):
                    totalUp += 1
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] in range(frameHeight // 2 - 30, frameHeight // 2 + 30):
                    totalDown += 1
                    to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        # text = "ID {}".format(objectID)
        # cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        # cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv.putText(frame, text, (10, frameHeight - ((i * 20) + 20)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# Process inputs
winName = 'object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"

if (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                            (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    cv.line(frame, (0, frameHeight // 2), (frameWidth, frameHeight // 2), (0, 255, 255), 2)

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes

    vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)