
This project involves using YOLO for counting people.
Moreover,as YOLO needs high resource,  tried combinig dlib's correlation tracker with YOLO and following are the compared results.


|     Detection Algorithm    |           Accuracy           |TimeComplexity | Inference Time| Result |
| -------------------------- | ---------------------------- |-------------- | ------------- |--------|
|             Yolo           |           Correct            |   Very slow   |   900    secs | <a href="https://github.com/angel-stha/YoloPeopleCounterwithdlib/blob/master/videos/example_01_yolo.avi">Yolo Result</a>                                           |
|         Yolo + dlib        |    Not as correct as Yolo    |      Slow     |   733.99 secs |<a href="https://github.com/angel-stha/YoloPeopleCounterwithdlib/blob/master/videos/yolo%2Bdlib_without_skipping.avi">YOLO + dlib result without skipping</a>      |
|Yolo + dlib (skipping frame)|      Moderately correct      |      Fast     |   143.40 sec  |<a href="https://github.com/angel-stha/YoloPeopleCounterwithdlib/blob/master/yolo%2Bdlib.avi">Yolo + dlip with frame skipping</a>                                  |










