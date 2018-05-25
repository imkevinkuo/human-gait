import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import math
import time
import os.path
def displayImage(files, zaxis, zd):
    index = 0
    while True:
        print(index, zaxis[index], zd[index])
        img = files[index]
        #highlight
        img = cv2.resize(img, (352,512))
        cv2.imshow("", img)
        ##
        k = cv2.waitKey(0)
        if k == 32:
            print(lmx)
            print(smn)
            print(lmn)
        if k == 27:
            cv2.destroyAllWindows()
            break
        if k == 2555904:
            #this is where you draw graph
            if index+1 < len(files):
                index += 1;
        if k == 2424832:
            if index > 0:
                index -= 1;
        if k == ord('s'):
            cv2.imwrite("out.jpg", img)
            print("saved")
def highlightPoints(img, points, color):
    temp = img.copy()
    for point in points:
        temp[point] = color
    return temp
def getImage(ind):
    name = str(ind).zfill(8) + ".png"
    img = cv2.imread(directory + "\\" + name, -1)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    return img
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
    return(angle)
    #return(center.astype(int),evecs.astype(int))
#get head's center of mass to sort between images with equal highestwhite vals
#find the neck with rightmost/leftmost particles on left/right sides
#takes previous values and does a regional search
def getHead(img, front, back):
    #(row,col)
    front = (front[0], 0)
    back = (back[0], 10000)
    r1 = 10
    r2 = 30
    if (front[0] > 0 and back[0] > 0):
        r1 = back[0] - 3
        r2 = front[0] + 3
    for row in range(r1, r2):
        for col in range(0, img.shape[1]-1): #front
            if np.all(img[row,col] >= [254,254,254]): 
                if col > front[1]:
                    front = (row,col)
                elif col == front[1] and row < front[0]:
                    front = (row,col)
                break
        for col in range(img.shape[1]-1, 0, -1): #back
            if np.all(img[row,col] >= [254,254,254]):
                if col < back[1]:
                    back = (row,col)
                elif col == back[1] and row > back[0]:
                    back = (row,col)
                break
    ##
##    for col in range(0, front[1]): #front
##        img[front[0], col] = [0,0,255]
##    for col in range(back[1], img.shape[1]-1): #back
##        img[back[0], col] = [0,0,255]
    #cv2.line(img, (front[1],front[0]), (back[1],back[0]), [0,0,255], 1)
    #above connecting line
    ztotal = 0
    znum = 0
    for row in range(0, back[0]):
        for col in range(0, img.shape[1]-1): #front
            if np.all(img[row,col] >= [254,254,254]):
                #img[row,col] = [0,0,255]
                ztotal += row
                znum += 1
    for row in range(back[0], front[0]):
        for col in range(0, img.shape[1]-1):
            if np.all(img[row,col] == [0,0,255]):
                break
            elif np.all(img[row,col] >= [254,254,254]):
                #img[row,col] = [0,0,255]
                ztotal += row
                znum += 1
    #return front/back neck xy, get center of mass y
    return (front, back, -(ztotal/znum))
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
########################
##### COMMAND LINE #####
########################
dirs = []
if len(sys.argv) == 1:
    for dirname, dirnames, filenames in os.walk('TreadmillDatasetA'):
    # print path to all subdirectories first.
        for subdirname in dirnames:
            name = os.path.join(dirname, subdirname)
            if name.endswith("gallery_10km"):
                dirs.append(name)
if len(sys.argv) >= 2:
    imindex = 1
    directory = sys.argv[1] # ENTER IN PATH TO CONTAINING FOLDER
    dirs.append(directory)
for directory in dirs:
    path = '.\\' + directory
    num_files = len([f for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path, f))])
    #speed = int(directory[-3]) #km/h
    #if directory[-5] == '_':
    #    speed = int(directory[-4:-2])
    speed = 10
    #print(directory)
    #origfiles = []
    files = []
    zaxis = []
    front = (0,0)
    back = (0,10000)
    for ind in range(0, num_files):
        img = getImage(ind+1)
        front, back, zavg = getHead(img, front, back)
        #fvecs = ellipse(img)
        #cv2.line(img, tuple(fvecs[0]), tuple(fvecs[0]+fvecs[1][0]), [0,0,255], 1)
        #cv2.line(img, tuple(fvecs[0]), tuple(fvecs[0]+fvecs[1][1]), [0,0,255], 1)
        zaxis.append(zavg)
        #zaxis.append(fvecs)
        files.append(img)
    z = [a for a in zaxis]
    zd = [a for a in zaxis]
    zd = diff(z)
    #plt.subplot(211)
    #plt.plot(z, 'bo', z, 'k')
    #plt.subplot(212)
    #plt.plot(zd, 'bo', zd, 'k')
    #plt.show()
    lmx = localMaxs(z, zd)
    lmn = localMins(z, zd)
    smn = singleMins(zd,lmx) # foot strike
    havg = sum(z[lmx[i]]-z[smn[i]] for i in range(1,len(smn)))/(len(smn)-1)
    clen = sum(lmn[f+1]-lmn[f] for f in range(1, len(lmn)-1))/(len(lmn)-2)
    #print(havg) #max height above ground
    #print(clen) #frames/stride, m/strike = clen*speed/216
    # compare ellipse features of similar frames,
    # e.g. when leg strikes or when highest point
    es = [ellipse(files[m]) for m in lmx]
    print(sum(es)/len(smn))
    #displayImage(files, zaxis, zd)
