import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import math
import os.path
import json

def displayImage(files, zaxis, zd):
    index = 0
    while True:
        sys.stdout.write("%d, %f, %f     \r" % (index, zaxis[index], zd[index]))
        sys.stdout.flush()
        cv2.imshow("", cv2.resize(files[index], (352,512)))
        k = cv2.waitKeyEx(0)
        if k == 32:
            print("Local maxes:", lmx) # high point
            print("Single mins:", smn) # foot strike
            print("Local mins:", lmn)  # low point
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit(0)
        if k == 2555904 and index+1 < len(files):
                index += 1;
        if k == 2424832 and index > 0:
                index -= 1;
        if k == ord('s'):
            cv2.imwrite("out.jpg", img)
            print("saved out.jpg")
def highlightPoints(img, points, color):
    temp = img.copy()
    for point in points:
        temp[point] = color
    return temp

# graph/math helpers
def ezdiff(lis): # difference between each index
    return [lis[i+1]-lis[i] for i in range(len(lis)-1)]
def diff(lis):
    dlis = []
    dlis.append(lis[1] - lis[0])
    for x in range(1, len(lis)-1):
        dlis.append(((lis[x] - lis[x-1])+(lis[x+1] - lis[x]))/2)
    dlis.append(lis[len(lis)-1] - lis[len(lis)-2])
    return dlis
# find local max of z graph to get avg cycle length
def localMaxs(lis, dlis):
    lm = []
    for x in range(1, len(lis)-4):
        sz = 5
        if x < 4:
            sz = x+1
        for r in range(1,sz):
            if (lis[x] < lis[x+r] or lis[x] < lis[x-r]):
                break
        if r == sz-1:
            lm.append(x)
    # prevent duplicates
    toRemove = set()
    for x in range(0, len(lm)-1):
        tol = 8 # tolerance
        if lm[x+1] - lm[x] < tol: # 2 maxes less than X frames apart
            if abs(dlis[lm[x]]) > abs(dlis[lm[x+1]]):
                toRemove.add(lm[x])
            else:
                toRemove.add(lm[x+1])
    for frame in toRemove:
        lm.remove(frame)
    return lm
def localMins(lis, dlis):
    lm = []
    for x in range(1, len(lis)-4):
        sz = 5
        if x < 4:
            sz = x+1
        for r in range(1,sz):
            if (lis[x] > lis[x+r] or lis[x] > lis[x-r]):
                break
        if r == sz-1:
            lm.append(x)
    # prevent duplicates
    toRemove = set()
    for x in range(0, len(lm)-1):
        tol = 8 # tolerance
        if lm[x+1] - lm[x] < tol: # 2 mins less than X frames apart
            if abs(dlis[lm[x]]) > abs(dlis[lm[x+1]]):
                toRemove.add(lm[x])
            else:
                toRemove.add(lm[x+1])
    for frame in toRemove:
        lm.remove(frame)
    return lm
# greater z value = lower irl
def singleMins(lis, lmax): # will search in each cycle for one min
    lm = []
    for x in range(0, len(lmax)-1):
        m = 10000
        mi = -1
        for f in range(lmax[x], lmax[x+1]):
            if lis[f] < m:
                m = lis[f]
                mi = f
        lm.append(mi);
    return lm
def plot(z, zd):
    plt.subplot(211)
    plt.plot(z, 'bo', z, 'k')
    plt.subplot(212)
    plt.plot(zd, 'bo', zd, 'k')
    plt.show()
    
def ellipse(img):
    #center
    xt = 0
    yt = 0
    c = 0
    points = set()
    for row in range(0, img.shape[0]):
        for col in range(0, img.shape[1]):
            if np.all(img[row,col] >= [254,254,254]):
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
def getNeck(img,front,back): # just get shortest distance across silhouette
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
        while firstW < img.shape[1] and np.any(img[row, firstW]) == 0:
            firstW += 1
        if firstW < img.shape[1]:
            col = firstW
            while col < img.shape[1]:
                if np.any(img[row,col]) != 0: # non-black pixel
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
    # draw connecting line and shade all pixels above
    cv2.line(img, (front[1],front[0]), (back[1],back[0]), [0,0,255], 1)
    ztotal = 0
    znum = 0
    for row in range(0, front[0]):
        for col in range(0, img.shape[1]):
            if np.all(img[row,col] == [0,0,255]):
                break
            elif np.any(img[row,col] != 0):
                img[row,col] = [0,0,255]
                ztotal += row
                znum += 1
    if znum == 0:
        return 0
    return -(ztotal/znum)

def usage():
    print("Usage: image.py <directory> <0 (head) or 1 (ellipse)>")
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
        img = cv2.imread(path + "\\" + name, -1)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
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
            cv2.line(img, tuple(center), tuple(center+evecs[0]), [0,0,255], 1)
            cv2.line(img, tuple(center), tuple(center+evecs[1]), [0,0,255], 1)
            data.append(angle)
        files.append(img)
    print(path + " loaded." + " "*20)
    return files, data
## COMMAND LINE ##
if len(sys.argv) == 2:
    T = sys.argv[1]
    parent_dir = ".\\TreadmillDatasetA"
    folder = "gallery_10km" 
    exclude = ["00116", "00117", "00124", "00128", "00134", "00140"]
    if T == "0" or T == "1":
        all_data = []
        subjects = os.listdir(parent_dir)
        for subject in subjects:
            if T == "1" or subject not in exclude:
                path = parent_dir + "\\" + subject + "\\" + folder
                files, data = processSubject(path, T)
                all_data.append(data)
        with open(folder + '.json', 'w') as outfile:
            json.dump(all_data, outfile)
if len(sys.argv) == 3: # one subject, will display video frames
    files, data = processSubject('.\\' + sys.argv[1], sys.argv[2])
    if sys.argv[2] == "0": # displaying data for head center method
        y = [y for y in data]
        yd = diff(y)
        plot(y,yd)
        lmx = localMaxs(y, yd)
        lmn = localMins(y, yd)
        smn = singleMins(yd,lmx)
        vert = sum(y[f] for f in lmx)/len(lmx) - sum(y[f] for f in lmn)/len(lmn)
        clen = sum(lmn[f+1]-lmn[f] for f in range(1, len(lmn)-1))/(len(lmn)-2)
        print("Vertical range:", vert, "pixels")
        print("Average stride length:", clen, "frames")
        print("Frame, Y-Average, Y-Velocity")
        displayImage(files, y, yd)

    if sys.argv[2] == "1":
        plot(data,data)
        displayImage(files,data,data)

    print()
    
else:
    usage()
