from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,Adam
from keras.utils import np_utils
import pygetwindow as gw
import winsound



from keras import backend as K
if K.backend() == 'tensorflow':
    import tensorflow
else:
    import theano


K.set_image_data_format('channels_first')
	
	
import numpy as np
import os
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
import speech_recognition as sr
from pynput.keyboard import Key, Controller
import os
import cv2
import matplotlib
from matplotlib import pyplot as plt

keyboard = Controller()
# input image dimensions
img_rows, img_cols = 200, 200

# number of channels
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
img_channels = 1


# Batch_size to train
batch_size = 32

## Number of output classes (change it accordingly)
nb_classes = 7

# Number of epochs to train (change it accordingly)
nb_epoch = 15  #25  3

# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3

#%%
#  data
path = "./"
path1 = "./gestures"    #path of folder of images

## Path2 is the folder which is fed in to training model
path2 = './imgfolder_b'


WeightFileName = []

# outputs
output = ["OK", "LEFT", "NOTHING", "PEACE", "PUNCH", "RIGHT", "STOP"]
# output = ["Fist", "Nothing", "Scissor", "Stop", "Thumbs Up"]
#output = ["PEACE", "STOP", "THUMBSDOWN", "THUMBSUP"]
guessed = 1
together_count = 0
playing=0
windowActivated = 'Google Chrome'
jsonarray = {}

#%%
def update(plot):
    global jsonarray
    h = 450
    y = 30
    w = 45
    font = cv2.FONT_HERSHEY_SIMPLEX

    #plot = np.zeros((512,512,3), np.uint8)
    
    #array = {"OK": 65.79261422157288, "NOTHING": 0.7953541353344917, "PEACE": 5.33270463347435, "PUNCH": 0.038031660369597375, "STOP": 28.04129719734192}
    
    for items in jsonarray:
        mul = (jsonarray[items]) / 100
        #mul = random.randint(1,100) / 100
        cv2.line(plot,(0,y),(int(h * mul),y),(255,0,0),w)
        cv2.putText(plot,items,(0,y+5), font , 0.7,(0,255,0),2,1)
        y = y + w + 30

    return plot

#%% For debug trace
def debugme():
    import pdb
    pdb.set_trace()

#%%
# Coloured to Grayscale while copying images from path1 to path2
def convertToGrayImg(path1, path2):
    listing = os.listdir(path1)
    for file in listing:
        if file.startswith('.'):
            continue
        img = Image.open(path1 +'/' + file)
        #img = img.resize((img_rows,img_cols))
        grayimg = img.convert('L')
        grayimg.save(path2 + '/' +  file, "PNG")

#%%
def modlistdir(path, pattern = None):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        #This check is to ignore any hidden files/folders
        if pattern == None:
            if name.startswith('.'):
                continue
            else:
                retlist.append(name)
        elif name.endswith(pattern):
            retlist.append(name)
    return retlist


# Load CNN model
def loadCNN(bTraining = False):
    global get_output
    model = Sequential()
    
    
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                        padding='valid',data_format='channels_first',
                        input_shape=(img_channels, img_rows, img_cols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    # Model summary
    model.summary()
    # Model conig details
    model.get_config()
    
    if not bTraining :
        #List all the weight files available in current directory
        WeightFileName = modlistdir('.','.hdf5')
        if len(WeightFileName) == 0:
            return 0
        else:
            print('Found these weight files - {}'.format(WeightFileName))
        #Load pretrained weights
        w = 0#int(input("Which weight file to load (enter the INDEX of it, which starts from 0): "))
        print(WeightFileName)
        fname = WeightFileName[int(w)]
        print("loading ", fname)
        model.load_weights(fname)

    # refer the last layer here
    layer = model.layers[-1]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    
    return model


def doAction(command):
    global windowActivated
    if command=="Activate":
        windowActivated=myCommand()
        try:
            handle = gw.getWindowsWithTitle(windowActivated)[0]
            handle.activate()
        except:
            print(gw.getAllTitles())
            print('Program not found')
            windowActivated = doAction("Activate");
    if command=="Quit":
        print(gw.getAllTitles())
        handle = gw.getWindowsWithTitle(windowActivated)[0]
        handle.activate()
        handle.maximize()
        keyboard.press(Key.alt.value)
        keyboard.press(Key.f4)
        keyboard.release(Key.f4)
        keyboard.release(Key.alt.value)
    
    if command=="Rewind":
        keyboard.press(Key.left)
        keyboard.release(Key.left)

    if command=="Forward":
        keyboard.press(Key.right)
        keyboard.release(Key.right)

    if command=="Next":
        handle = gw.getWindowsWithTitle(windowActivated)[0]
        handle.activate()
        keyboard.press(Key.down)
        keyboard.release(Key.down)

    if command=="Back":
        handle = gw.getWindowsWithTitle(windowActivated)[0]
        handle.activate()
        keyboard.press(Key.up)
        keyboard.release(Key.up)
    

def myCommand(param="Listening for command:"):    
    "listens for commands"
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(param)
        r.pause_threshold = 0.2
        r.non_speaking_duration=0.1
        r.energy_threshold = 250
        # r.adjust_for_ambient_noise(source, duration=1)
        frequency = 14000  
        duration = 300 
        winsound.Beep(frequency, duration)
        audio = r.listen(source)
    
    try:       
        command = r.recognize_google(audio).lower()
        print('You said: ' + command + '\n')
    
    #loop back to continue to listen for commands if unrecognizable speech is received
    except sr.UnknownValueError:
        frequency = 7000  
        duration = 300  
        winsound.Beep(frequency, duration) 
        print('....')
        command = myCommand(param);
    
    return command.capitalize()

# This function does the guessing work based on input images
def guessGesture(model, img):
    global output, get_output, jsonarray, guessed, together_count, playing, windowActivated
    image = np.array(img).flatten()
    image = image.reshape(img_channels, img_rows,img_cols)
    image = image.astype('float32') 
    image = image / 255
    rimage = image.reshape(1, img_channels, img_rows, img_cols)
    
    prob_array = get_output([rimage, 0])[0]
    
    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1
    
    # Get the output with maximum probability
    import operator
    
    guess = max(d.items(), key=operator.itemgetter(1))[0]
    guessednew = output.index(guess)
    prob  = d[guess]
    if guessed==guessednew:
        together_count+=1
    else:
        together_count=0
    if prob > 60.0 and together_count==5:
        print(guess + "  Probability: ", prob)
        if guess=="STOP":
            handle = gw.getWindowsWithTitle(windowActivated)[0]
            handle.activate()
            keyboard.press(Key.space)
            keyboard.release(Key.space)
        if guess=="PUNCH":
            handle = gw.getWindowsWithTitle(windowActivated)[0]
            handle.activate()
            keyboard.press('m')
            keyboard.release('m')
        if guess=="RIGHT":
            handle = gw.getWindowsWithTitle(windowActivated)[0]
            handle.activate()
            keyboard.press(Key.right)
            keyboard.release(Key.right)
        if guess=="LEFT":
            handle = gw.getWindowsWithTitle(windowActivated)[0]
            handle.activate()
            keyboard.press(Key.left)
            keyboard.release(Key.left)
        if guess=="PEACE":
            handle = gw.getWindowsWithTitle(windowActivated)[0]
            handle.activate()
            if playing==0:
                playing=1
                print("Launching VLC Media Player....")
                os.startfile("paradise.webm")
                # os.startfile("AI Review-1.pptx")
            else:
                playing=0
                print("Closing VLC Media Player....")
                keyboard.press(Key.ctrl)
                keyboard.press('q')
                keyboard.release(Key.ctrl)
                keyboard.press('q')
        if guess=="OK":
            keyboard.press('f')
        jsonarray = d
        guessed=guessednew       
        return output.index(guess)

    else:
        guessed=guessednew
        return 2


def initializers():
    imlist = modlistdir(path2)
    
    image1 = np.array(Image.open(path2 +'/' + imlist[0])) # open one image to get size
    #plt.imshow(im1)
    
    m,n = image1.shape[0:2] # get the size of the images
    total_images = len(imlist) # get the 'total' number of images
    
    # create matrix to store all flattened images
    immatrix = np.array([np.array(Image.open(path2+ '/' + images).convert('L')).flatten()
                         for images in sorted(imlist)], dtype = 'f')
    

    
    print(immatrix.shape)
    
    input("Press any key")
    

    label=np.ones((total_images,),dtype = int)
    
    samples_per_class = int(total_images / nb_classes)
    print("samples_per_class - ",samples_per_class)
    s = 0
    r = samples_per_class
    for classIndex in range(nb_classes):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class
    
    '''
    # eg: For 301 img samples/gesture for 4 gesture types
    label[0:301]=0
    label[301:602]=1
    label[602:903]=2
    label[903:]=3
    '''
    
    data,Label = shuffle(immatrix,label, random_state=2)
    train_data = [data,Label]
     
    (X, y) = (train_data[0],train_data[1])
     
     
    # Split X and y into training and testing sets
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
     
    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
     
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
     
    # normalize
    X_train /= 255
    X_test /= 255
     
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test



def trainModel(model):

    # Split X and y into training and testing sets
    X_train, X_test, Y_train, Y_test = initializers()

    # Now start the training of the loaded model
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                 verbose=1, validation_split=0.2)

    ans = input("Do you want to save the trained weights - y/n ?")
    if ans == 'y':
        filename = input("Enter file name - ")
        fname = path + str(filename) + ".hdf5"
        model.save_weights(fname,overwrite=True)
    else:
        model.save_weights("newWeight.hdf5",overwrite=True)
        
    visualizeHis(hist)

    # Save model as well
    # model.save("newModel.hdf5")
#%%

def visualizeHis(hist):
    # visualizing losses and accuracy
    keylist = hist.history.keys()
    #print(hist.history.keys())
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    
    #Tensorflow new updates seem to have different key name
    if 'acc' in keylist:
        train_acc=hist.history['acc']
        val_acc=hist.history['val_acc']
    else:
        train_acc=hist.history['accuracy']
        val_acc=hist.history['val_accuracy']
    xc=range(nb_epoch)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    plt.show()

#%%
def visualizeLayers(model):
    print("Are you here?")
    imlist = modlistdir('./imgs')
    if len(imlist) == 0:
        print('Error: No sample image file found under \'./imgs\' folder.')
        return
    else:
        print('Found these sample image files - {}'.format(imlist))

    img = int(input("Which sample image file to load (enter the INDEX of it, which starts from 0): "))
    #layerIndex = int(input("Enter which layer to visualize. Enter -1 to visualize all layers possible: "))
    layerIndex = int(input("Enter which layer to visualize:"))
    
    if img <= len(imlist):
        
        image = np.array(Image.open('./imgs/' + imlist[img]).convert('L')).flatten()
        
        ## Predict
        print('Guessed Gesture is {}'.format(output[guessGesture(model,image)]))
        
        # reshape it
        image = image.reshape(img_channels, img_rows,img_cols)
        
        # float32
        image = image.astype('float32')
        
        # normalize it
        image = image / 255
        
        # reshape for NN
        input_image = image.reshape(1, img_channels, img_rows, img_cols)
    else:
        print('Wrong file index entered !!')
        return
    
    
    
        
    # visualizing intermediate layers
    #output_layer = model.layers[layerIndex].output
    #output_fn = theano.function([model.layers[0].input], output_layer)
    #output_image = output_fn(input_image)
    
    if layerIndex >= 1:
        visualizeLayer(model,img,input_image, layerIndex)
    else:
        tlayers = len(model.layers[:])
        print("Total layers - {}".format(tlayers))
        for i in range(1,tlayers):
             visualizeLayer(model,img, input_image,i)

#%%
def visualizeLayer(model, img, input_image, layerIndex):

    layer = model.layers[layerIndex]
    
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    activations = get_activations([input_image, 0])[0]
    output_image = activations
    
    
    ## If 4 dimensional then take the last dimension value as it would be no of filters
    if output_image.ndim == 4:
        # Rearrange dimension so we can plot the result
        #o1 = np.rollaxis(output_image, 3, 1)
        #output_image = np.rollaxis(o1, 3, 1)
        output_image = np.moveaxis(output_image, 1, 3)
        
        print("Dumping filter data of layer{} - {}".format(layerIndex,layer.__class__.__name__))
        filters = len(output_image[0,0,0,:])
        
        fig=plt.figure(figsize=(8,8))
        # This loop will plot the 32 filter data for the input image
        for i in range(filters):
            ax = fig.add_subplot(6, 6, i+1)
            #ax.imshow(output_image[img,:,:,i],interpolation='none' ) #to see the first filter
            ax.imshow(output_image[0,:,:,i],'gray')
            #ax.set_title("Feature map of layer#{} \ncalled '{}' \nof type {} ".format(layerIndex,
            #                layer.name,layer.__class__.__name__))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        plt.tight_layout()
        #plt.show()
        savedfilename = "img_" + str(img) + "_layer" + str(layerIndex)+"_"+layer.__class__.__name__+".png"
        fig.savefig(savedfilename)
        print("Create file - {}".format(savedfilename))
        #plt.close(fig)
    else:
        print("Can't dump data of this layer{}- {}".format(layerIndex, layer.__class__.__name__))


