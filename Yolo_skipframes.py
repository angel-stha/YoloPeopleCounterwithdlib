from yolo.centroidtracker import CentroidTracker
from yolo.trackableobject import TrackableObject
import argparse
import cv2
import numpy as np
import os.path
import dlib
import csv
import time
import imutils
from imutils.video import VideoStream
from imutils.video import FPS

def Argument_Parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False,
                    help="path to Caffe 'deploy' prototxt file", default="./mobilenet_ssd/MobileNetSSD_deploy.prototxt")
    ap.add_argument("-m", "--model", required=False,
                    help="path to Caffe pre-trained model", default="./mobilenet_ssd/MobileNetSSD_deploy.caffemodel")
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

def Detector(args):
    with open('coco.names', 'rt') as f:
        CLASSES = f.read().rstrip('\n').split('\n')
        # print(len(classes))

        modelConfiguration = "darknet/cfg/yolov3.cfg";
        modelWeights = "darknet/yolov3.weights";

        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        # net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

        if not args.get("input", False):
            print("[INFO] starting video stream...")
            #vs = VideoStream("rtsp://rtspflow0430:tBH^&`7r@QY2(rD{@116.91.197.77/axis-media/media.amp").start()
            # vs = cv2.VideoCapture('rtsp://rtspflow0430:tBH^&`7r@QY2(rD{@116.91.197.77/axis-media/media.amp')
            # vs = VideoStream(src = '0'.start())
            time.sleep(2.0)

        # otherwise, grab a reference to the video file
        else:
            print("[INFO] opening video file...")
            vs = cv2.VideoCapture(args["input"])

        if not os.path.exists(args.get("datapath")):
            print("Creating a new csv file")
            with open(args.get("datapath"), 'w') as csvfile:
                wrt = csv.writer(csvfile)
                wrt.writerow(["Metric", "Time", "IN/OUT", "Total In", "Total Out"])
        # initialize the video writer (we'll instantiate later if need be)
        writer = None

        W = None
        H = None

        # instantiate our centroid tracker, then initialize a list to store
        # each of our dlib correlation trackers, followed by a dictionary to
        # map each unique object ID to a TrackableObject
        ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        trackers = []
        trackableObjects = {}

        totalFrames = 0
        totalDown = 0
        totalUp = 0
        left = 0
        right = 0
        # start the frames per second throughput estimator
        fps = FPS().start()
        while True:
            # grab the next frame and handle if we are reading from either
            # VideoCapture or VideoStream
            frame = vs.read()
            frame = frame[1] if args.get("input", False) else frame

            # if we are viewing a video and we did not grab a frame then we
            # have reached the end of the video
            if args["input"] is not None and frame is None:
                break

            # cv2.imshow("frame",frame)
            # resize the frame to have a maximum width of 500 pixels (the
            # less data we have, the faster we can process it), then convert
            # the frame from BGR to RGB for dlib
            frame = imutils.resize(frame, width=500)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # if the frame dimensions are empty, set them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # if we are supposed to be writing a video to disk, initialize
            # the writer
            if args["output"] is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                         (W, H), True)

            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
            status = "Waiting"
            rects = []

            # check to see if we should run a more computationally expensive
            # object detection method to aid our tracker
            if totalFrames % args["skip_frames"] == 0:
                # set the status and initialize our new set of object trackers
                status = "Detecting"
                trackers = []
                (frameHeight, frameWidth) = frame.shape[:2]

                # convert the frame to a blob and pass the blob through the
                # network and obtain the detections
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
                net.setInput(blob)
                detections = net.forward()
                print(detections)
                # loop over the detections
                for detection in detections:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    classIds = []
                    confidences = []
                    boxes = []
                    if confidence > 0.6:
                        print(detection)
                        dete = (detection[1], detection[2], detection[3], detection[0])
                        # det.append(dete)
                        center_x = int(detection[1] * frameWidth)
                        print(center_x)
                        center_y = int(detection[2] * frameHeight)
                        print(center_y)
                        width = int(detection[3] * frameWidth)
                        print(width)
                        height = int(detection[0] * frameHeight)
                        print(height)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        classIds.append(classId)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])

            # Perform non maximum suppression to eliminate redundant overlapping boxes with
            # lower confidences.

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)
            for i in indices:
                print(i[0])
                i = i[0]
                box = boxes[i]
                print(box)
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(left, top, width, height)
                print(rect)
                print(rgb)
                tracker.start_track(rgb, rect)
                # Class "person"
                if classIds[i] == 0:
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # add the bounding box coordinates to the rectangles list
                    rects.append((startX, startY, endX, endY))

                    # use the centroid tracker to associate the (1) old object
                    # centroids with (2) the newly computed object centroids
                    objects = ct.update(rects)
                    counting(objects)

            # otherwise, we should utilize our object *trackers* rather than
            # object *detectors* to obtain a higher frame processing throughput
            else:
                # loop over the trackers
                for tracker in trackers:
                    # set the status of our system to be 'tracking' rather
                    # than 'waiting' or 'detecting'
                    status = "Tracking"

                    # update the tracker and grab the updated position
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # add the bounding box coordinates to the rectangles list
                    rects.append((startX, startY, endX, endY))

            # draw a horizontal line in the center of the frame -- once an
            # object crosses this line we will determine whether they were
            # moving 'up' or 'down'
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
            cv2.line(frame, (W // 2, H), (W // 2, 0), (0, 255, 255), 2)

            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            objects = ct.update(rects)

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
                    x = [c[0] for c in to.centroids]
                    vertical = centroid[0] - np.mean(x)

                    # check to see if the object has been counted or not
                    if not to.counted:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        if direction < 0 and centroid[1] < H // 2:
                            totalUp += 1
                            with open(args.get("datapath"), "a", newline='') as csvfile:
                                row = ["tc", str(time.strftime("%Y/%m/%d %H:%M:%S")), "Out", totalDown, totalUp]
                                data_dict = {'Metric': "TC", 'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                             'In/Out': "OUT", 'Total In': totalUp, 'Total Out': totalDown}
                                # row = [datetime.now(), "Out", totalDown, totalUp, totalDown - totalUp]
                                # mqttc.publish(topic, json.dumps(data_dict))
                                wrt = csv.writer(csvfile)
                                wrt.writerow(row)
                            # writer.write
                            to.counted = True

                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object
                        elif direction > 0 and centroid[1] > H // 2:
                            totalDown += 1
                            with open(args.get("datapath"), "a", newline='') as csvfile:
                                # row = [datetime.now(), "IN", totalDown, totalUp, totalDown - totalUp]
                                row = ["tc", str(time.strftime("%Y/%m/%d %H:%M:%S")), "In", totalDown, totalUp]
                                data_dict = {'Metric': "TC", 'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                             'In/Out': "In", 'Total In': totalUp, 'Total Out': totalDown}
                                # mqttc.publish(topic, json.dumps(data_dict))
                                wrt = csv.writer(csvfile)

                                wrt.writerow(row)

                            to.counted = True
                        elif vertical < 0 and centroid[0] < W // 2:
                            left += 1
                            with open(args.get("datapath"), "a", newline='') as csvfile:
                                row = ["tc", str(time.strftime("%Y/%m/%d %H:%M:%S")), "Out", totalDown, totalUp]
                                data_dict = {'Metric': "TC", 'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                             'In/Out': "OUT", 'Total In': totalUp, 'Total Out': totalDown}
                                # row = [datetime.now(), "Out", totalDown, totalUp, totalDown - totalUp]
                                # mqttc.publish(topic, json.dumps(data_dict))
                                wrt = csv.writer(csvfile)
                                wrt.writerow(row)
                            # writer.write
                            to.counted = True

                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object
                        elif vertical > 0 and centroid[0] > W // 2:
                            right += 1
                            with open(args.get("datapath"), "a", newline='') as csvfile:
                                # row = [datetime.now(), "IN", totalDown, totalUp, totalDown - totalUp]
                                row = ["tc", str(time.strftime("%Y/%m/%d %H:%M:%S")), "In", totalDown, totalUp]
                                data_dict = {'Metric': "TC", 'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                             'In/Out': "In", 'Total In': totalUp, 'Total Out': totalDown}
                                # mqttc.publish(topic, json.dumps(data_dict))
                                wrt = csv.writer(csvfile)

                                wrt.writerow(row)

                            to.counted = True

                # store the trackable object in our dictionary
                trackableObjects[objectID] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("Out ", totalUp),
                ("In", totalDown),
                ("Left", left),
                ("Right", right),
                ("Status", status),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # check to see if we should write the frame to disk
            if writer is not None:
                writer.write(frame)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # increment the total number of frames processed thus far and
            # then update the FPS counter
            totalFrames += 1
            fps.update()

        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # check to see if we need to release the video writer pointer
        if writer is not None:
            writer.release()

        # if we are not using a video file, stop the camera video stream
        if not args.get("input", False):
            vs.stop()

        # otherwise, release the video file pointer
        else:
            vs.release()
        # close any open windows
        cv2.destroyAllWindows()

        # Continue the network loop forever
        # mqttc.loop_forever()

    def Message_Broker():
        # Publish a message
        mqttc.publish(topic, "my message")

    if __name__ == "__main__":
        ap = Argument_Parser()
        Detector(ap)
        print('ok')