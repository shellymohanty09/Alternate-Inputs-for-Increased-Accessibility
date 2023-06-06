import cv2
import numpy as np
import os
import time

import threading

import gestureCNN as myNN

minValue = 70

x0 = 400
y0 = 200
height = 200
width = 200

saveImg = False
guessGesture = False
myCommand = False
visualize = False

kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

binaryMode = True
bkgrndSubMode = False
mask = 0
bkgrnd = 0
counter = 0
numOfSamples = 803
gestname = ""
path = ""
mod = 0

banner =  '''\nWhat would you like to do ?
    1- Recognise gestures and control desktop
    2- Voice recognition
    3- Train the model 
    4- Visualize feature maps of different layers of trained model
    5- Exit  
    '''


#%%
def saveROIImg(img):
    global counter, gestname, path, saveImg
    if counter > (numOfSamples - 1):
        saveImg = False
        gestname = ''
        counter = 0
        return
    
    counter = counter + 1
    name = gestname + str(counter)
    print("Saving img:",name)
    cv2.imwrite(path+name + ".png", img)
    time.sleep(0.04 )


#%%
def skinMask(frame, x0, y0, width, height, framecount, plot):
    global guessGesture, visualize, mod, saveImg
    # HSV values
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_range, upper_range)
    
    mask = cv2.erode(mask, skinkernel, iterations = 1)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)
    
    mask = cv2.GaussianBlur(mask, (15,15), 1)
    
    res = cv2.bitwise_and(roi, roi, mask = mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True and (framecount % 5) == 4:
        t = threading.Thread(target=myNN.guessGesture, args = [mod, res])
        t.start()
    elif visualize == True:
        layer = int(input("Enter which layer to visualize "))
        cv2.waitKey(0)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False
    
    return res


def binaryMask(frame, x0, y0, width, height, framecount, plot ):
    global guessGesture, visualize, mod, saveImg
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True and (framecount % 5) == 4:
        t = threading.Thread(target=myNN.guessGesture, args = [mod, res])
        t.start()
    elif visualize == True:
        layer = int(input("Enter which layer to visualize "))
        cv2.waitKey(1)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False

    return res

def bkgrndSubMask(frame, x0, y0, width, height, framecount, plot):
    global guessGesture, takebkgrndSubMask, visualize, mod, bkgrnd, saveImg
        
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
    if takebkgrndSubMask == True:
        bkgrnd = roi
        takebkgrndSubMask = False
        print("Refreshing background image for mask...")		

    diff = cv2.absdiff(roi, bkgrnd)

    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
    mask = cv2.GaussianBlur(diff, (3,3), 5)
    mask = cv2.erode(diff, skinkernel, iterations = 1)
    mask = cv2.dilate(diff, skinkernel, iterations = 1)
    res = cv2.bitwise_and(roi, roi, mask = mask)
    
    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True and (framecount % 5) == 4:
        t = threading.Thread(target=myNN.guessGesture, args = [mod, res])
        t.start()
        
    elif visualize == True:
        layer = int(input("Enter which layer to visualize "))
        cv2.waitKey(0)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False
    
    
    return res
	
	
	
def Main():
    global guessGesture, visualize, mod, binaryMode, bkgrndSubMode, mask, takebkgrndSubMask, x0, y0, width, height, saveImg, gestname, path
    quietMode = False
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.5
    fx = 10
    fy = 350
    fh = 18

    while True:
        ans = int(input( banner))
        if ans == 1:
            mod = myNN.loadCNN()
            break
        elif ans == 2:
            mod = myNN.loadCNN()
            break
        elif ans == 3:
            mod = myNN.loadCNN(True)
            myNN.trainModel(mod)
            input("Press any key to continue")
            break
        elif ans == 4:
            if not mod:
                mod = myNN.loadCNN()
            else:
                print("Will load default weight file")
            
            myNN.visualizeLayers(mod)
            input("Press any key to continue")
            continue
        else:
            print("Thank You have a Nice Day!!!")
            return 0


    cap = cv2.VideoCapture(0)
    # cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    ret = cap.set(3,640)
    ret = cap.set(4,480)

    framecount = 0
    fps = ""
    start = time.time()

    plot = np.zeros((512,512,3), np.uint8)
    
    while(True):
        ret, frame = cap.read()
        max_area = 0
        
        frame = cv2.flip(frame, 3)
        frame = cv2.resize(frame, (640,480))                    
        if ret == True:
            if bkgrndSubMode == True:
                roi = bkgrndSubMask(frame, x0, y0, width, height, framecount, plot)
            elif binaryMode == True:
                roi = binaryMask(frame, x0, y0, width, height, framecount, plot)
            else:
                roi = skinMask(frame, x0, y0, width, height, framecount, plot)

            
            framecount = framecount + 1
            end  = time.time()
            timediff = (end - start)
            if( timediff >= 1):
                fps = 'FPS:%s' %(framecount)
                start = time.time()
                framecount = 0

        cv2.putText(frame,fps,(10,20), font, 0.7,(0,255,0),2,1)
        cv2.putText(frame,'Options:',(fx,fy), font, 0.7,(0,255,0),2,1)
        cv2.putText(frame,'b - Toggle Binary/SkinMask',(fx,fy + 3*fh), font, size,(0,255,0),1,1)
        #cv2.putText(frame,'v - Toggle Voice Recognition Mode',(fx,fy + 3*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'g - Toggle Gesture Recognition Mode',(fx,fy + 5*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'ESC - Exit',(fx,fy + 7*fh), font, size,(0,255,0),1,1)
        
        if not quietMode:
            cv2.imshow('Original',frame)        
            cv2.imshow('ROI', roi)

            if guessGesture == True:
                plot = np.zeros((512,512,3), np.uint8)
                plot = myNN.update(plot)
            
            cv2.imshow('Gesture Probability',plot)
        key = cv2.waitKey(5) & 0xff
        if key == 27:
            break
        elif key == ord('b'):
            binaryMode = not binaryMode
            bkgrndSubMode = False
            if binaryMode:
                print("Binary Threshold filter active")
            else:
                print("SkinMask filter active")
        elif key == ord('x'):
            takebkgrndSubMask = True
            bkgrndSubMode = True
            print("BkgrndSubMask filter active")
        elif key == ord('g'):
            guessGesture = not guessGesture
            print("Gesture Recognition Mode - {}".format(guessGesture))
        elif key == ord('v'):
            myCommand = True
            print("Voice Recognition Mode - {}".format(myCommand))
            commanded=myNN.myCommand()
            myNN.doAction(commanded)
        elif key == ord('i'):
            y0 = y0 - 5
        elif key == ord('k'):
            y0 = y0 + 5
        elif key == ord('j'):
            x0 = x0 - 5
        elif key == ord('l'):
            x0 = x0 + 5
        elif key == ord('q'):
            quietMode = not quietMode
            print("Quiet Mode - {}".format(quietMode))
        elif key == ord('s'):
            saveImg = not saveImg
            
            if gestname != '':
                saveImg = True
            else:
                print("Enter a gesture group name first, by pressing 'n'")
                saveImg = False
        elif key == ord('n'):
            gestname = input("Enter the gesture folder name: ")
            try:
                os.makedirs(gestname)
            except OSError as e:
                # if directory already present
                if e.errno != 17:
                    print('Some issue while creating the directory named -' + gestname)
            
            path = "./"+gestname+"/"
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()

