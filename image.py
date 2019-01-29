import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import math
import os.path
import json

def displayImage(files, data):
    index = 0
    while True:
        sys.stdout.write("%d, %f     \r" % (index, data[index]))
        sys.stdout.flush()
        cv2.imshow("", cv2.resize(files[index],(352,512)))
        k = cv2.waitKeyEx(0)
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit(0)
        if k == 2555904 and index+1 < len(files):
            index += 1;
        if k == 2424832 and index > 0:
            index -= 1;
def diff(lis):
    dlis = []
    dlis.append(lis[1] - lis[0])
    for x in range(1, len(lis)-1):
        dlis.append(((lis[x] - lis[x-1])+(lis[x+1] - lis[x]))/2)
    dlis.append(lis[len(lis)-1] - lis[len(lis)-2])
    return dlis
def plot(z):
    plt.subplot(211)
    plt.plot(z, 'bo', z, 'k')
    plt.show()
def ellipse(img):
    #center
    xt = 0
    yt = 0
    c = 0
    points = set()
    for row in range(0, img.shape[0]):
        for col in range(0, img.shape[1]):
            if img[row,col] > 250:
                xt += col
                yt += row
                c += 1
                points.add((col,row))
    xm = xt/c
    ym = yt/c
    center = np.array([xm,ym])
    # xm and ym are the avg coordinates
    cxx = sum([(p[0] - xm)*(p[0] - xm) for p in points])/(c-1)
    cyy = sum([(p[1] - ym)*(p[1] - ym) for p in points])/(c-1)
    cxy = sum([(p[0] - xm)*(p[1] - ym) for p in points])/(c-1)
    cov = [[cxx, cxy],[cxy, cyy]]
    eig = np.linalg.eig(cov)
    evals = eig[0]
    evecs = eig[1]
    angle = math.asin(evecs[1][1])
    for e in range(0,2):
        ax = math.sqrt(5.99*evals[e])
        evecs[e] = evecs[e]*ax
        evecs[e][1] = -evecs[e][1]
    return (center.astype(int),evecs.astype(int),angle)
def getNeck(img,front,back): # shortest distance across silhouette
    cutoffA = 10
    cutoffB = img.shape[0]//3
    if front != None and back != None:
        cutoffA = back[0]-2
        cutoffB = front[0]+2
    fronts = []
    backs = []
    for row in range(cutoffA, cutoffB):
        firstW = 0
        lastW = 0
        while firstW < img.shape[1] and img[row, firstW] < 3:
            firstW += 1
        if firstW < img.shape[1]:
            col = firstW
            while col < img.shape[1]:
                if img[row,col] > 250: # non-black pixel
                    lastW = col
                col += 1
        fronts.append((row,firstW))
        backs.append((row,lastW))
    return closestPair(fronts,backs)
def closestPair(A,B):
    min_A = A[0]
    min_B = B[0]
    min_d = dist(A[0], B[0])
    for a in A:
        for b in B:
            d = dist(a,b)
            if d < min_d:
                min_A = a
                min_B = b
                min_d = d
    return min_A, min_B
def dist(a,b):
    dx = abs(a[0]-b[0])
    dy = abs(a[1]-b[1])
    return dx*dx + dy*dy
# Uses the 'front' and 'back' pixels from getNeck to
# calcualte geometric center of the subject's head
def getHeadCenter(img, front, back):
##    cv2.line(img, (front[1],front[0]), (back[1],back[0]), [0,0,255], 1)
    m = (front[1]-back[1])/(front[0]-back[0]+0.001)
    b = back[1] - m*back[0]
    # row is x, col is y
    ztotal = 0
    znum = 0
    for row in range(0, front[0]):
        y = m*row + b
        for col in range(0, img.shape[1]):
            if col < y and img[row,col] > 250:
                ztotal += row
                znum += 1
    if znum == 0:
        return 0
    return -(ztotal/znum)

def usage():
    print("Usage: image.py <directory> <extract type>")
    sys.exit()

def processSubject(path, t): # t is type of feature extraction
    num_files = len([f for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path, f))])
    files, data = [], []
    front, back = None, None
    print()
    for ind in range(0, num_files):
        # read in file
        name = str(ind+1).zfill(8) + ".png"
        img = cv2.imread(path + "\\" + name, 0)
        # update progress
        size = 40
        bars = ((ind+1)*size)//num_files # 10 bars total
        sys.stdout.write(name + " [" + u"\u25A0"*bars + " "*(size-bars) + "] " + '\r')
        sys.stdout.flush()
        if t == "0": # head center method
            front, back = getNeck(img, front, back)
            zavg = getHeadCenter(img, front, back)
            data.append(zavg)
        if t == "1": # ellipse method
            center, evecs, angle = ellipse(img)
##            cv2.line(img, tuple(center), tuple(center+evecs[0]), [0,0,255], 1)
##            cv2.line(img, tuple(center), tuple(center+evecs[1]), [0,0,255], 1)
            data.append(angle)
        if t == "2": # count # of white pixels
            data.append(int(np.sum(img > 250)))
        files.append(img)
    print(path + " loaded." + " "*20)
    return files, data
def avg_shad(files, data):
    for f in files:
        f[f < 10] = 0
        f[f > 245] = 255
    avg_imgs = []
    m = split_cyc(data)
    for i in range(0,len(m)-2):
        N = m[i+2]-m[i]
        avg_imgs.append(np.uint8(sum([(a/N) for a in files[m[i]:m[i+2]]])))
    return avg_imgs
def split_cyc(s):
    cutoff = min(s) + (max(s)-min(s))/2
    under = [i for i in range(len(s)) if s[i] < cutoff]
    i = 0
    mins = []
    for u in range(len(under)-1):
        if under[u+1]-under[u] > 3:
            mins.append(min(under[i:u+1], key=lambda f:s[f]))
            i = u+1
    if len(mins)%2 == 0:
        mins = mins[:-1]
    return mins
## COMMAND LINE ##
if len(sys.argv) == 2:
    T = sys.argv[1]
    cwd = os.getcwd()
    datadir = "TreadmillDatasetA"
    folder = "probe_10km"
    accepted = ["0", "1", "2"]
    if T in accepted:
        exclude = []
        if T == "0":
            exclude = ["00116", "00117", "00124", "00128", "00134", "00140"]
##        if T == "2":
##            os.mkdir(datadir+"_avg")
        all_data = []
        subjects = os.listdir(os.path.join(cwd,datadir))
        for subject in subjects:
            if T != "0" or subject not in exclude:
                path = os.path.join(cwd, datadir, subject, folder)
                files, data = processSubject(path, T)
                all_data.append(data)
                if T == "2":
##                    os.mkdir(os.path.join(datadir+"_avg", subject))
                    os.mkdir(os.path.join(datadir+"_avg", subject, folder))
                    av_f = avg_shad(files,data)
                    av_p = os.path.join(cwd, datadir+"_avg", subject, folder)
                    for i in range(len(av_f)):
                        filepath = os.path.join(av_p, str(i) + ".png")
                        cv2.imwrite(filepath, av_f[i])
        with open(folder + '.json', 'w') as outfile:
            json.dump(all_data, outfile)
elif len(sys.argv) == 3: # one subject, will display video frames
    files, data = processSubject('.\\' + sys.argv[1], sys.argv[2])
    plot(data)
    displayImage(files,data)
    print()
    
else:
    usage()
