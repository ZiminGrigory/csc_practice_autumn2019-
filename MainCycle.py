import sys
sys.path.append('dan_tf/DAN_V2')
sys.path.append('classifier')  
sys.path.append('keras-yolo3') 

from  My_DAN_V2 import storeFramesWithMarks, storeEyesData, storeEyesDataAndImages, showFramesWithMarksAndPredictRes
from my_web_predict import  createCameraGenerator, createCameraGeneratorWithCount, folderGenerator, createVideoGenerator
from classifier import  getClassifier, createPredictor


# stored my face on disk with landmarks
# gen = createCameraGenerator(112,68);
# storeFramesWithMarks(gen)

# stored csv with open eyes data just points x1,y1,x2,y2,... (no distance)
#gen = createCameraGeneratorWithCount(112, 68, 500)
## storeEyesData(gen, "/home/greg/dev/csc_practice_autumn2019/300W_HELEN/eyesOpen.csv")
# storeEyesData(gen, "/home/greg/dev/csc_practice_autumn2019/300W_HELEN/eyesClose.csv")


# camera video result -- uncomment
# gen = createCameraGenerator(112,68);
# clsPredict = createPredictor(getClassifier('classifier/my_dumped_classifier2.pkl'))
# showFramesWithMarksAndPredictRes(gen, clsPredict)

#video
gen = createVideoGenerator(112,68, '/home/greg/Videos/salma.mp4');
clsPredict = createPredictor(getClassifier('classifier/my_dumped_classifier2.pkl'))
showFramesWithMarksAndPredictRes(gen, clsPredict)




# gen = folderGenerator(112, 68, '/home/greg/dev/csc_practice_autumn2019/facesForClassifier/images/2/')
# storeEyesDataAndImages(gen, '/home/greg/dev/csc_practice_autumn2019/facesForClassifier/images/2_res/')
