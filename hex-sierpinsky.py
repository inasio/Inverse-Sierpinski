import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput
from time import time

def plot_triangle(x,y):
    #side = 2*np.sqrt((x[1]-x[0])**2 + (y[1]-y[0])**2)
    #side = 1 - np.sqrt(x[1]**2 + y[1]**2)
    side = 0.2
    pb.plot(x,y,'k',linewidth=side)
    pb.plot(x[2::-2],y[2::-2],'k',linewidth=side)

def plot_triangle_inverse(x,y,params):
    a,b = params
    norms = (abs(x)**a + abs(y)**a)**b
    x = x/norms
    y = y/norms
    x0 = np.concatenate([x, [x[0]]])
    y0 = np.concatenate([y, [y[0]]])
    pb.plot(y0,x0,'k',linewidth=0.4)

def midpoint(x1, x2):
	return [(x1[0] + x1[1])/2., (x2[0] + x2[1])/2.]

def reproduce_triangles(triangles):
    X, Y, Z = triangles.shape
    counter = 0
    new_triangles = np.zeros([2, 4, Z*3])
    for i in xrange(Z):
        x = triangles[0,:3,i]
        y = triangles[1,:3,i]

        midp01 = midpoint(x[0:2], y[0:2])
        midp02 = midpoint(x[0::2], y[0::2])
        midp12 = midpoint(x[1:], y[1:])
        new_triangles[:,:3,counter]   = np.array([[x[0], midp01[0], midp02[0]],[y[0], midp01[1], midp02[1]]])
        new_triangles[:,3, counter]   = x[0], y[0]
        new_triangles[:,:3,counter+1] = np.array([[x[1], midp12[0], midp01[0]],[y[1], midp12[1], midp01[1]]])
        new_triangles[:,3, counter+1] = x[1], y[1]
        new_triangles[:,:3,counter+2] = np.array([[x[2], midp02[0], midp12[0]],[y[2], midp02[1], midp12[1]]])
        new_triangles[:,3, counter+2] = x[2], y[2]
        counter += 3
        #new_triangles[1,:3,counter] = y[0], midp01[1], midp02[1]
        #new_triangles[1,:,:] = [np.array([x[1], midp12[0], midp01[0]]), np.array([y[1], midp12[1], midp01[1]])]
        #new_triangles[2,:,:] = [np.array([x[2], midp02[0], midp12[0]]), np.array([y[2], midp02[1], midp12[1]])]
    return new_triangles

if __name__=="__main__":


    # Start with a single triangle
    #x = np.zeros(3)
    #y = np.zeros(3)
    #for i in range(3):
        #num = np.pi/2 + 2*np.pi*i/3.
        #x[i] = np.cos(num)
        #y[i] = np.sin(num)
    #triangles = []
    #triangles.append([x,y])
        
    # Start with a hexagon split into six triangles
    triangles = np.zeros([2,4,6])
    for i in range(6):
    # The figure is shifted in y by h in order to center the figure 
    # in an empty space and avoid div by 0, looks better.
        h=np.sin(np.pi/3)*.5*1.5
        #x = [0, np.cos(2*np.pi*i/6), np.cos(2*np.pi*(i+1)/6)]
        #y = [h, h+np.sin(2*np.pi*i/6), h+np.sin(2*np.pi*(i+1)/6)]
        triangles[0,:3,i] = 0, np.cos(2*np.pi*i/6), np.cos(2*np.pi*(i+1)/6)
        triangles[1,:3,i] = h, h+np.sin(2*np.pi*i/6), h+np.sin(2*np.pi*(i+1)/6)
        triangles[0,3,i] = triangles[0,0,i]
        triangles[1,3,i] = triangles[1,0,i]

    plt.plot(triangles[0,:,:], triangles[1,:,:])
    tr2 = reproduce_triangles(triangles)
    plt.plot(tr2[0,:,:], tr2[1,:,:])

    plt.show()
    caca
    # key parameter: how many generations of triangles in the fractal
    generation = 1
    params = [0.45, 0.65]
    #params = [1.25, 1.95]
    for i in xrange(generation):
        counter = 0
        num_triangles = 6*3**(i+1)
        new_triangles = np.zeros([2,4,num_triangles])
        #for tr in triangles:
        for j in xrange(num_triangles):
        #    tr = 
            temp_triangles = reproduce_triangle(tr)
            for temp_tr in temp_triangles:
                new_triangles.append(temp_tr)
                counter += 1
        print i*1./generation
        triangles = new_triangles

    #Uncomment for uninverted Sierpinski fractal
    #plt.figure('siers normal')
    #plt.clf()
    #for t in triangles:
    #    x = np.concatenate([t[0],[t[0][0]]])
    #    y = np.concatenate([t[1],[t[1][0]]])
    #    plt.plot(x,y)

    t3 = time()
    plt.figure('sierps')
    plt.clf()
     
    # Uncomment for profiling details 
    #with PyCallGraph(output=GraphvizOutput()):
    #    for t in triangles:
    #        plot_triangle_inverse(t[0], t[1], params)

    for t in triangles:
        plot_triangle_inverse(t[0], t[1], params)

    t4 = time()
    print 't4 ', t4-t3
    plt.show()
    #        #for i in newts:
    #            #plot_triangle(newts[i][0], newts[i][1])

    #        #plot_triangle(x,y)
    #        plt.plot(0,0,'r.')
    #        plt.text(0,0,str(a)+', '+str(b), color='blue')
    #        ax = plt.gca()
    #        ax.autoscale(tight=True)
    #        lx,rx = ax.get_xlim()
    #        ly,hy = ax.get_ylim()
    #        xmax = max(abs(lx), abs(rx))
    #        ymax = max(abs(ly), abs(hy))
    #        plt.axis([-xmax, xmax, -ymax, ymax])
    #        #plt.axis([-1, 1, -1, 1])
    #        #plt.axis('equal')
    #        plt.axis('off')
    #        counter_figs += 1
    #        #pb.savefig('gif_figs/inverse_sierpinski'+str(counter_figs).zfill(4)+'.png',dpi = 300)
    #        #pb.savefig('inverse_sierpinski2.pdf',dpi = 1200)
    #        #pb.savefig('inverse_sierpinski2.raw',dpi = 1200)
    #        #pb.savefig('inverse_sierpinski2.svg',dpi = 1200)
    #        plt.show()
    #        plt.pause(0.1)
