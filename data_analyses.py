"""
To do, sorted by priority:
#Find FFT method: understand what the frequency axis and intensity axis mean.
For sine the frequency is right, but is it good for the actual data?
#Normalize intensity axis. It works for sine but for actual data highest peak is very small

#find an optimum between interpolation types, smoothing (averaging) and FFT results
#Look into improving smoothing of data. Is there a better method than taking N point avg

#Let python create a folder if it does not exist yet.
#Is there a way to remove/mask unwanted data points in python?

"""
import numpy as np
from scipy.interpolate import interp1d
import scipy.fftpack
import matplotlib.pyplot as plt
import glob

#Which data colums do you want to analyse, start counting at 0.
p = 0   #x value
q = 1   #y value

#Minimum/maximum x values?
xmin = 0
xmax = 10

#Do you want to take an N point average? With what N? How many times?
take_avg = True
N = 4
M = 5

#What degree polynomal do you want fitted?
deg=2

#Data in 1/B? Necessary for FFT analyses
inv_B = False

#Do you want to interpolate, and what type? (Linear, cubic)
interp = True
interpolate_type='linear'
pick_no_of_interp_points = False             #If true use interp_points * nr of data points
interp_points = 10000                          #as number of interpolation points.


#Do you want to perform an FFT? And do you want to plot it and/or write to file?
fft = True
fmin = 0                            #Give min frequency in T
fmax = 20                        #Give max frequency in T
plot_fft = True
write_fft_to_file = False


#=============output:=============#
#Want to plot the actual data?
plot_real_data = False
#Plot the data with a subtracted background?
plot_subtracted_data = True

#Create offset? handy if there are many files plotted over each other.
create_offset = False
offset = 0.00001
offset_increment = 0.00001

#Want to write to a file?
################IN PROGRESS####################
write_to_file = False

#path = 'C:/Users/bvlaar/surfdrive/Programming/Python-projects/background-subtraction/data/'
path = 'C:/Users/bvlaar/surfdrive/Programming/Python-projects/'
file_list = ['sine.dat'
            #'11052018_cell4.002.dat',
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

def write_fft(xf, yf, path, filename):
    fpo = open(path+'/fft/'+filename[:len(filename)-4]+'_fft.dat', 'w+')
    fpo.write('frequency (kT) \t Intensity \n')
    
    for i in range(len(xf)):
        fpo.write(str(xf[i]) + '\t' + str(2.0/len(x)*abs(yf[:len(x)//2])) + '\n')
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
                if i>= N/2 and i+N/2<=len(x):               #Make sure that you can take N/2 
                    x_temp += x[i+j-int(N/2)]               #points around (above and below) 
                    y_temp += y[i+j-int(N/2)]               #a point i.
            x_avg[i]=x_temp/N
            y_avg[i]=y_temp/N
    return x_avg, y_avg

#Interpolation
def interpolate(x,y,kind):
    f = interp1d(x,y, kind=kind)
    return f
    
#Fourier Transformation
def fft_data(x,y):
    #Number of samplepoints:
    N=len(x)
    #Sample point spacing:
    T=max(x)/N
    xf = np.linspace(0.0,1./(2.*T),N/2)
    yf = scipy.fftpack.fft(y)
    return xf,yf


#============================Main: Call the functions============================#
if (take_avg):
    print('Taking ', N, 'point average, repeating', M, 'times.')
if(create_offset and (plot_subtrachted_data or plot_real_data)):
    print('Creating an offset of', offset_increment)
if(inv_B):
    print('Using 1/B')
if(interp):
        print('Interpolating with', interpolate_type, 'interpolation')
if(write_to_file):
    print('Writing to files')
if(fft):
    print('Performing FFT on the data')
    if(plot_fft):
        print('Plotting FFT')
    if(write_fft_to_file):
        print('Writing FFT data to file')


for filename in file_list:
    """
    For each filename in file_list: load header, x and y.
    if take_avg:
        -> average data M times over N points.
    Fit the data with a polynomial of degree deg.
    Subtract the polynomial from the data.
    if data is periodic in 1/B and you want to do FFT
        -> x = 1/x
    if data needs to be evenly spaced (interpolated)
        -> Interpolate to x_int, y_int by a linear or cubic interpolation.
    if FFT and not interpolated:
        -> Give a warning and try
    """

    ############Read data from file, store in header, x, y############
    header, x, y = read_file(path+filename, p, q)
    xlabel = header[p]
    x = np.array(x)
    y = np.array(y)
    x_data = x                                              #Storing the data for comparisson
    y_data = y                                              #with analysis.
    
    ############Take an N points average of x and y:############
    if (take_avg):
        for i in range(M):      #Repeat the averaging M times.
            x,y = Npointavg(x,y,N)
            x = x[int(N/2+1):(len(x)-int(N/2))]
            y = y[int(N/2+1):len(y)-int(N/2)]
            #y_fit = fit_function(x,y,deg)                  #Uncomment to see how taking
            #plt.scatter(x,y-y_fit, marker = '+', s=10)     #multiple averages changes data.
            #plt.plot(x,y-y_fit)                            #y is fitted after each average.           
            

    ############fit the background############
    y_fit = fit_function(x, y, deg)
    y_data_fit = fit_function(x_data, y_data, deg)          #Keep track of the data for comparisson

    
    ############remove the background############
    y = y - y_fit        
    y_data_subtracted = y_data - y_data_fit                 #keep track of the data for comparisson

    ############Use 1/B instead of B.############
    if(inv_B):
        xlabel = 'inverse field (1/T)'
        for i in range(len(x)):
            if x[i]!=0:
                x[i] = 1.0/x[i]
        for i in range(len(x_data)):                        #Also change the x data to 1/B to
            if x_data[i]!=0:                                #compare the data with the averaged
                x_data[i] = 1./x_data[i]                    #and interpolated plots

    ############Interpolate (x,y)############       
    if(interp):
        if(not pick_no_of_interp_points):
            no_of_interp_points = len(x)
        else:
            no_of_interp_points = interp_points*len(x)      #Take interp_points * no of data points 
                                                            #as number of interpolation points
        x_int = np.linspace(x[0], x[len(x)-1], no_of_interp_points)
        y_int = interpolate(x,y, interpolate_type)

        #Uncomment to plot interpolated data in a scatterplot. 
        #plt.xlim(min(x_new), max(x_new))                   #Needed to scale axes when a scatter plot
        #dy = (max(y) - min(y))*0.1                         #is used. Known issue in matplotlib.
        #plt.ylim(min(y)-dy,max(y)+dy)                      #
        #plt.scatter(x_new, y_int(x_new), marker='+', s=3, label = 'interpolate '+interpolate_type)

    
    ############Plotting the analysed data############
    if(plot_real_data):
        plt.plot(x_data,y_data, label = filename, lw=0.5)
        
    if(plot_subtracted_data):
        if(create_offset):
            y = y+offset
            offset += offset_increment

        #Create a figure in which the (averaged) subtracted data, and/or interpolated data can be plotted in.
        plt.figure(1)
        #Plot the data with a background subtracted
        #plt.plot(x_data, y_data, label = filename, linewidth=0.5, color='r')
        
        #Plot average data with a background subtracted
        plt.plot(x, y, label = str(M)+'*'+str(N)+'point average of '+filename, linewidth=0.5)

        #Plot the interpolation of data with a subtracted background.
        if(interp):
            plt.plot(x_int,y_int(x_int), label = interpolate_type+' interpolation', linewidth=0.5)
        
            ############Uncomment to plot interpolated data in a scatterplot############
            #plt.xlim(min(x_int), max(x_int))    #Needed to scale axes when a scatter plot
            #dy = (max(y) - min(y))*0.1          #is used. Known issue in matplotlib.
            #plt.ylim(min(y)-dy,max(y)+dy)       #
            #plt.scatter(x_int, y_int(x_int),s=3, facecolor='0.0', marker='+', label = 'interpolate '+interpolate_type)
            #plt.legend()
        plt.legend()

        
    #Writing to file
    if(write_to_file):
        write_file(header,x_int,y_int(x_int),path,filename)


    if(fft):
        if( not interp):
            print('Data is not interpolated and might not be evenly spaced, output is possibly unreliable!')
        xf,yf = fft_data(x_int,y_int(x_int))
        if(plot_fft):
            plt.figure(2)
            xlabel = 'Frequency'
            ylabel = 'Intensity'
            plt.plot(xf,2.0/len(x_int)*abs(yf[:len(x_int)//2]),label = 'FFT of '+filename)
        if(write_fft_to_file):
            write_fft(xf,yf,path,filename)
                            
        
    
#============================Plotting============================#
    
if((plot_real_data or plot_subtracted_data) or (fft and plot_fft)):
    plt.xlabel(xlabel)
    #plt.ylabel('torque (a.u.)')
    plt.legend()
    if(fft and plot_fft):
        plt.xlim(fmin, fmax)
    plt.show()


