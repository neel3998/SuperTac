import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from numpy import arange
import math
from .mapData import config_3Ddata

class CoordinateDesc():
    def __init__(self, pat = 'Waves', num=4):
        self.pat = pat
        self.num = num 
        self.precision = 1
        
        self.specimenWidth = 36 # x-direction
        self.specimenLength = 108 # y-direction


    def get_Z_coord(self,pt):
        #where points is x,y points of which we have to calculate z and c is the function
        x,y=pt[0], pt[1]
        if self.pat=="Waves":
            y_i,y_f,A,phi, num_amps = config_3Ddata(self.pat, self.num)
            if y<=y_i or y>=y_f:
                z=0
            else:
                z=A*math.sin((y-y_i)*np.pi/((y_f-y_i)/num_amps) +phi*math.pi/180)
            return z

        if self.pat=="Ridges":
            y_i,y_f,A,B,C = config_3Ddata(self.pat, self.num)
            # for i in self.points:
            if y<=y_i or y>=y_f:
                z=0
            else:
                cent=[] #appending all centres of 
                for k in range(int((y_f-y_i-(B/2))/C)+1):
                    cent.append(y_i+(B/2)+k*C)
                #  print(cent)
                nearest_cent=1e8
                ind=1e8
                for j in cent:
                    if abs(y-j)<=ind:
                        ind=abs(y-j)
                        nearest_cent=j
                
                #  print(nearest_cent)
                    if y-nearest_cent>=0:
                        z=-(2*A/B)*(y-(nearest_cent+B/2))
                    else:
                        z=(2*A/B)*(y-(nearest_cent-B/2))

                    if z<=0:
                        z = 0
            return z

        if self.pat=="Bumps":
            y_i,y_f,A,B = config_3Ddata(self.pat, self.num)
            if y<=y_i or y>=y_f:
                z=0
            else:
                cent=[] #appending all centres of 
                for k in range(int((y_f-y_i-(A))/B)+1):
                    cent.append(y_i+(A)+k*B)
                nearest_cent=1e8
                ind=1e8
                for j in cent:
                    if abs(y-j)<=ind:
                        ind=abs(y-j)
                        nearest_cent=j
                z_sq=(A**2-(y-nearest_cent)**2)
                if z_sq<=0:
                    z = 0
                else:
                    z = math.sqrt(z_sq)

            return z

        if self.pat=="Blob": 
            y_i,y_f,A,B_x,B_y = config_3Ddata(self.pat, self.num)
            if y<=y_i or y>=y_f:
                z=0
            else:
                centres = []
                for i in range(self.num):
                    x0 = B_x/2
                    y0 = y_i + A + B_y*i
                    for j in range(self.num):
                        xCen = x0 + j*B_x
                        yCen = y0
                        centres.append([xCen,yCen])   
                
                closest_cent = centres[0]
                inCircle = 0
                for centre in centres:
                    temp = np.linalg.norm([centre[0] - x, centre[1] - y])
                    if temp<=A:
                        inCircle = 1
                        closest_cent = centre
                        break
                
                if inCircle:
                    z = math.sqrt(A**2 - (closest_cent[0] - x)**2 - (closest_cent[1] - y)**2)
                else:
                    z = 0
                
            return z
        
        if self.pat == 'plain':
            return 0 

if __name__=='__main__':
    a=CoordinateDesc()
    a.pat = "Blob"
    z = a.get_Z_coord([4.5, 40])
    print(z)
