#To do:
#Look into improving smoothing of data. Is there a better method than taking N point avg
#Let python create a folder if it does not exist yet.
#Is there a way to remove/mask unwanted data points in python?
#Interpolate function
#Find best FFT method
#(extra: Create a better legend with just the filenames)


import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import glob

#Which data colums do you want to analyse, start counting at 0.
p = 2   #x value
q = 7   #y value

#Minimum/maximum x values?
xmin = 30.0
xmax = 36.0

#Do you want to take an N point average? With what N? How many times?
take_avg = False
N = 3
M = 1

#What degree polynomal do you want fitted?
deg=2

#Do you want to interpolate, and what type? (Linear, cubic)
interpolate = True
interpolate_type='linear'


#Plotting:
#Want to plot the data?
plot_data = True

#Create offset?
create_offset = False
offset = 0.00001
offset_increment = 0.00001

#Data in 1/B? Necessary for FFT analyses
inv_B = False

#Want to write to a file?
################IN PROGRESS####################
write_to_file = False

path = 'C:/Users/bvlaar/surfdrive/Programming/Python-projects/background-subtraction/data/'
file_list = ['11052018_cell4.002.dat',
             #'11052018_cell4.006.dat',
             #'11052018_cell4.009.dat',
             #'11052018_cell4.012.dat',
             #'12052018_cell4.001.dat',
             #'12052018_cell4.005.dat',
             #'12052018_cell4.007.dat',
             #'12052018_cell4.010up.dat',
             #'12052018_cell4.012.dat',
             #'12052018_cell4.015.dat',
             #'12052018_cell4.017.dat'
             ]



#============================File handling functions============================#

#Read columns from file into x and y lists.
def read_file(filename, p, q):
    fpi = open(filename,'r')
    #Read first line as header and split on each tab.
    header = fpi.readline().split('\t')
    line = fpi.readline()
    counter = 0
    x = []
    y = []
    while(line):
        linesplit = line.split()
        if(float(linesplit[p]) >= xmin and float(linesplit[p]) <= xmax):
            x.append(float(linesplit[p]))
            y.append(float(linesplit[q]))
        line = fpi.readline()
    fpi.close()
    return header, x, y

def write_file(header, x,y, path, filename):
    #Open/create a file in subfolder: 'background_subtracted'
    fpo = open(path+'/background_subtracted/'+filename[:len(filename)-4]+'output.dat', 'w+')
    fpo.write(header[p]+'\t'+header[q]+'\n')
    if(len(x)!=len(y)):
       print('Error: Length x and y not equal')
    for i in range(len(x)):
       fpo.write(str(x[i]) + '\t' + str(y[i]) + '\n')
    fpo.close()

#============================Data manipulation============================#

#Fit (x,y) with a polynomial of degree 'deg'
#Creates a list y_fit with the fitted values.
def fit_function(x, y, deg):
    fit_coefs = np.polyfit(x, y, deg)
    y_fit = np.zeros(len(y))
    for i in range(len(x)):
        deg_temp = 0
        while(deg_temp<=deg):
            y_fit[i] += fit_coefs[deg_temp]*(x[i]**(deg-deg_temp))
            deg_temp += 1
    return y_fit

#Take an N point average around each point.
def Npointavg(x, y, N):
    if len(x)!=len(y):
        print('Error: Length of x and y is not equal!')
    else:
        x_avg = np.zeros(len(x))
        y_avg = np.zeros(len(y))
        for i in range(len(x)):
            x_temp = 0
            y_temp = 0
            for j in range(N):
                if i>= N/2 and i+N/2<=len(x): #Make sure that you can take N/2 points around (above and below) a point i.
                    x_temp += x[i+j-int(N/2)]
                    y_temp += y[i+j-int(N/2)]
            x_avg[i]=x_temp/N
            y_avg[i]=y_temp/N
    return x_avg, y_avg

#Interpolation
def interpolate(x,y,kind):
    f = interp1d(x,y, kind=kind)
    
    


#============================Main: Call the functions============================#
if (take_avg):
    print('Taking ', N, 'point average, repeating ', M, 'times.')
if(create_offset and plot_data):
    print('Creating an offset of ', offset_increment)
if(inv_B):
    print('Using 1/B')
if(write_to_file):
    print('Writing to files')


for filename in file_list:
    header, x, y = read_file(path+filename, p, q)
    xlabel = header[p]
    x = np.array(x)
    y = np.array(y)
    #Take an N points average:
    if (take_avg):
        for i in range(M):      #Repeat the averaging M times.
            x,y = Npointavg(x,y,N)
            x = x[int(N/2+1):(len(x)-int(N/2))]
            y = y[int(N/2+1):len(y)-int(N/2)]
            #y_fit = fit_function(x,y,deg)                  #Uncomment to see how taking 
            #plt.scatter(x,y-y_fit, marker = '+', s=10)     #multiple averages changes data.
            #plt.plot(x,y-y_fit)                            #

    #fit the background
    y_fit = fit_function(x, y, deg) 
    #remove the background
    y = y - y_fit

            
    #Use 1/B instead of B.  
    if(inv_B):
        xlabel = 'inverse field (1/T)'
        for i in range(len(x)):
            if x[i]!=0:
                x[i] = 1.0/x[i]

    #Interpolate (x,y)
    #if(interpolate):
        
    
    
    

    if(write_to_file):
        write_file(header,x,y,path,filename)

    #plt.plot(x,y)
    #plt.plot(x,y_fit)
    
    #Create offset in plot
    if(plot_data):
        if(create_offset):
            y = y+offset
            offset += offset_increment
        plt.plot(x, y, label = filename, linewidth=0.5)

    
#============================Plotting============================#
    
if(plot_data):
    plt.xlabel(xlabel)
    #plt.ylabel('torque (a.u.)')
    #plt.legend(bbox_to_anchor=(1, 1.), loc=2, borderaxespad=0.)
    #plt.legend()
    plt.show()


