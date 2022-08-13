import argparse
import cv2
import torch
import random
import torchvision.transforms as transforms
import numpy as np
import threading
import time
import requests
from playsound import playsound

SoundWarnFlag = False

PhoneWarnFlag = False
CigaretteWarnFlag = False
TextingWarnFlag = False
DistractionWarnFlag = False

PhoneWarnCounter = 0
CigaretteWarnCounter = 0
TextingWarnCounter = 0
DistractionWarnCounter = 0

PhoneWarnCounter2 = 0
CigaretteWarnCounter2 = 0
TextingWarnCounter2 = 0

Person = "Fatma"
FPS = 0

distractionLevel = 0
distLevPHN = 0.3
distLevCIG = 0.2
distLevTEX = 0.3
distLevLKG = 0.2

conveyor = []
value_conveyor = []



from elements.yolo import OBJ_DETECTION
from elements.Look_classifier import Look_Classifier
from elements.Lstm_decision import Lstm_decision


from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import scale_coords
from utils.plots import plot_one_box

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def Warn_Conveyor(warnType,person):
    sound_file = "./wav/" + warnType + " - " + person + ".wav"
    return sound_file

def play_conveyor(conveyor):
    global SoundWarnFlag
    if not SoundWarnFlag:
        SoundWarnFlag = True
        for sound in conveyor:
            playsound(sound)
        SoundWarnFlag = False

def Start_Warn(conv):
    global conveyor
    t1 = threading.Thread(target = play_conveyor, args = ([conv]), daemon = True)
    t1.start()
    conveyor = []
###############################
def PhoneWarnFlagToggle(t):
    global PhoneWarnFlag
    global PhoneWarnCounter
    global PhoneWarnCounter2
    PhoneWarnCounter +=1
    if PhoneWarnCounter > 3:
        PhoneWarnCounter = 0
        PhoneWarnCounter2 += 1
        Start_Warn([Warn_Conveyor("PhoneCallWarn", Person)])
    time.sleep(30)
    PhoneWarnFlag = False

def CigaretteWarnFlagToggle(t):
    global CigaretteWarnFlag
    global CigaretteWarnCounter
    global CigaretteWarnCounter2
    CigaretteWarnCounter +=1
    if CigaretteWarnCounter > 3:
        CigaretteWarnCounter = 0
        CigaretteWarnCounter2 += 1
        Start_Warn([Warn_Conveyor("CigaretteWarn",Person)])
    time.sleep(30)
    CigaretteWarnFlag = False

def TextingWarnFlagToggle(t):
    global TextingWarnFlag
    global TextingWarnCounter
    global TextingWarnCounter2
    TextingWarnCounter +=1
    if TextingWarnCounter > 3:
        TextingWarnCounter = 0
        TextingWarnCounter2 += 1
        Start_Warn([Warn_Conveyor("TextingWarn", Person)])
    time.sleep(30)
    TextingWarnFlag = False

def DistractionWarnFlagToggle(t):
    global DistractionWarnFlag
    global DistractionWarnCounter
    DistractionWarnCounter +=1
    #Start_Warn([Warn_Conveyor("DistractionLevel", Person)])
    time.sleep(30)
    DistractionWarnFlag = False
        
###############################
def dweet(variable):
    url = "https://dweet.io/dweet/for/"
    url += "DistLev"
    url += "?"
    url += "&".join(variable)
    try:
        #requests.get(url = url)
        requests.post(url=url)
    except:
        print("Hata: Dweet gönderilemedi! İnternet bağlantısını kontrol ediniz.")
###############################
def equalize_image(img,method = 'HE'):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    if method == 'HE':
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    elif method == 'CLAHE':
        clahe = cv2.createCLAHE(clipLimit = 5)
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])# + 30
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def detect():
    
    Object_detector = OBJ_DETECTION(opt.DetectorWeights, opt.device, opt.img_size)
    
    
    if opt.drv_gaze:
        Gaze_Detector = Look_Classifier(opt.DriverGazeWeights, "resnet50", opt.deviceGAZE)
    if opt.lstm_detect:
        LSTM_detector = Lstm_decision(opt.LSTMWeights[0],opt.deviceLSTM)
    
    
    names = Object_detector.classes
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    global PhoneWarnFlag
    global CigaretteWarnFlag
    global TextingWarnFlag
    global DistractionWarnFlag
    
    
    
    if opt.source == 0:
        cap = cv2.VideoCapture(0)
    else:
        print(opt.source)
        cap = cv2.VideoCapture(opt.source, cv2.CAP_GSTREAMER)
    #img = torch.zeros((1, 3, imgsz, imgsz))
    im0 = []
    if cap.isOpened():
        global distractionLevel
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            time1 = time.time()
            ret, frame = cap.read()
            img = letterbox(frame, new_shape=opt.img_size)[0]
            im0 = frame.copy()
            distractionLevel = 0.0
            if ret:
                # detection process
                objs,pred= Object_detector.detect(img)
                LstmDetFlag = True
                looking = ""
                # plotting
                conveyor = []
                phn_trd = threading.Thread(target = PhoneWarnFlagToggle, args = [time.time()],daemon=True)
                cgr_trd = threading.Thread(target = CigaretteWarnFlagToggle, args = [time.time()],daemon=True)
                tx_trd = threading.Thread(target = TextingWarnFlagToggle, args = [time.time()],daemon=True)
                dist_trd = threading.Thread(target = DistractionWarnFlagToggle, args = [time.time()],daemon=True)
                value_conveyor = []
                for obj in objs:
                    # print(obj)
                    label = obj['label']
                    score = obj['score']
                    [(xmin,ymin),(xmax,ymax)] = obj['bbox']
                    color = colors[names.index(label)]
                    #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    coords = torch.tensor([pred[0][:, :4][objs.index(obj)].tolist()])
                    coords = scale_coords(img.shape, coords, im0.shape).round()
                    plot_one_box(coords.tolist()[0], im0, label=label, color=color, line_thickness=1)
                    #im0 = cv2.rectangle(im0, (xmin,ymin), (xmax,ymax), color, 2) 
                    #im0 = cv2.putText(im0, f'{label} ({str(score)})', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA)
                    
                    if opt.drv_gaze:
                        if label == "DriverFace":
                            looking = Gaze_Detector.predict_DriverGaze(frame, obj)
                            #print(looking)
                if opt.lstm_detect and LstmDetFlag:
                    LstmDetFlag = False
                    lstm_dat = LSTM_detector.elestiem(pred,names,frame.shape,img.shape)
                    lstm_dat.append(1.0 if looking=="forward" else 0.0)
                    lstm_o,lstm_output = LSTM_detector.lstm_det(lstm_dat)
                    distractionLevel += lstm_output[0]*distLevPHN
                    distractionLevel += lstm_output[1]*distLevCIG
                    distractionLevel += lstm_output[2]*distLevTEX
                    distractionLevel += (1.0 if looking=="forward" else 0.0)*distLevLKG
                    lab = ''.join([str(elem) for elem in lstm_o])
                    PhoneCall = bool(int(lstm_o[0]))
                    Smoking = bool(int(lstm_o[1]))
                    Texting = bool(int(lstm_o[2]))
                    width = im0.shape[1]
                    height = im0.shape[0]
                    text_scale = height/960
                    txt_wid = int(width/4)
                    txt_height = int(20*(text_scale/0.5))
                    im0 = cv2.putText(im0, "Phone Call: " + ("True" if PhoneCall else "False"), (0, txt_height), 0, text_scale, ([0, 0, 255] if PhoneCall else [255, 0, 0]), thickness=1, lineType=cv2.LINE_AA)
                    im0 = cv2.putText(im0, "   Smoking: " + ("True" if Smoking else "False"), (txt_wid*1, txt_height), 0, text_scale, ([0, 0, 225] if Smoking else [225, 0, 0]), thickness=1, lineType=cv2.LINE_AA)
                    im0 = cv2.putText(im0, "   Texting: " + ("True" if Texting else "False"), (txt_wid*2, txt_height), 0, text_scale, ([0, 0, 225] if Texting else [225, 0, 0]), thickness=1, lineType=cv2.LINE_AA)
                    im0 = cv2.putText(im0, "   Looking: " + (looking),                          (txt_wid*3, txt_height), 0, text_scale, ([0, 0, 225] if looking=="other" else [225, 0, 0]), thickness=1, lineType=cv2.LINE_AA)
                    value_conveyor.append("PhnCall" + ("=1" if PhoneCall else "=0"))
                    value_conveyor.append("Smk" + ("=1" if Smoking else "=0"))
                    value_conveyor.append("Txt" + ("=1" if Texting else "=0"))
                    if PhoneCall and not PhoneWarnFlag:
                        conveyor.append(Warn_Conveyor("PhoneCall", Person))
                        PhoneWarnFlag = True
                        phn_trd.start()
                    if Smoking and not CigaretteWarnFlag:
                        conveyor.append(Warn_Conveyor("Cigarette", Person))
                        CigaretteWarnFlag = True
                        cgr_trd.start()
                    if Texting and not TextingWarnFlag:
                        conveyor.append(Warn_Conveyor("Texting", Person))
                        TextingWarnFlag = True
                        tx_trd.start()
                    if distractionLevel.item()*100 > 40.0 and not DistractionWarnFlag:
                        conveyor.append(Warn_Conveyor("DistractionLevel", Person))
                        DistractionWarnFlag = True
                        dist_trd.start()
                    

            
            if conveyor != []: Start_Warn(conveyor)
            im0 = cv2.putText(im0, "Distraction: % {:.2f}".format(distractionLevel.item()*100),(0, txt_height*3), 0, text_scale,[0, 0, 255], thickness=1, lineType=cv2.LINE_AA)
            value_conveyor.append("DistractionLevel={:.2f}".format(distractionLevel.item()*100))
            value_conveyor.append("PhoneWarnCounter={}".format(PhoneWarnCounter))
            value_conveyor.append("CigaretteWarnCounter={}".format(CigaretteWarnCounter))
            value_conveyor.append("TextingWarnCounter={}".format(TextingWarnCounter))
            value_conveyor.append("PhoneWarnCounter2={}".format(PhoneWarnCounter2))
            value_conveyor.append("CigaretteWarnCounter2={}".format(CigaretteWarnCounter2))
            value_conveyor.append("TextingWarnCounter2={}".format(TextingWarnCounter2))
            value_conveyor.append("DistractionWarnCounter={}".format(DistractionWarnCounter))
            
            threading.Thread(target = dweet, args = [value_conveyor], daemon = True).start()
            FPS = int(1/(time.time()-time1))
            im0 = cv2.putText(im0, "FPS: {}".format(FPS),(0, txt_height*2), 0, text_scale,[0, 0, 255], thickness=1, lineType=cv2.LINE_AA)
            cv2.imshow("CSI Camera", im0)
            keyCode = cv2.waitKey(30)
            if keyCode == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DetectorWeights', nargs='+', type=str, default='weights/ObjectDetectorModel.pt', help='model.pt path(s)')
    parser.add_argument('--DriverGazeWeights', nargs='+', type=str, default='weights/DriverGazeModel.pt', help='model.pt path(s)')
    parser.add_argument('--LSTMWeights', nargs='+', type=str, default='weights/LSTMmodel.pkl', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='csicam', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--deviceGAZE', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--deviceLSTM', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--extract-lstmdata', action='store_true', help='extract lstm data')    
    parser.add_argument('--lstm-detect', action='store_true', help='extract lstm data')    
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--drv-gaze', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    
    if opt.source == "csicam":
        opt.source = gstreamer_pipeline(
            capture_width=3264,
            capture_height=1848,
            display_width=1280,
            display_height=720,
            framerate=28,
            flip_method=6
            )
    elif opt.source == "webcam":
        opt.source = 0
    
    with torch.no_grad():
        detect()
