import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
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
    #side = 2*np.sqrt((x[1]-x[0])**2 + (y[1]-y[0])**2)
    #side = 1 - 0.5/np.sqrt(x[1]**2 + y[1]**2)
    side = 0.4
    x = x/norms
    y = y/norms
    #print x
    x0 = np.concatenate([x, [x[0]]])
    y0 = np.concatenate([y, [y[0]]])
    pb.plot(y0,x0,'k',linewidth=side)
    #pb.plot(x[2::-2],y[2::-2],'k',linewidth=side)

def midpoint(x1, x2):
	return [(x1[0] + x1[1])/2., (x2[0] + x2[1])/2.]

def reproduce_triangle(triangle):
    x, y = triangle
    new_triangles = np.zeros([3,2,3])
    midp01 = midpoint(x[0:2], y[0:2])
    midp02 = midpoint(x[0::2], y[0::2])
    midp12 = midpoint(x[1:], y[1:])
    new_triangles[0,:,:] = [np.array([x[0], midp01[0], midp02[0]]), np.array([y[0], midp01[1], midp02[1]])]
    new_triangles[1,:,:] = [np.array([x[1], midp12[0], midp01[0]]), np.array([y[1], midp12[1], midp01[1]])]
    new_triangles[2,:,:] = [np.array([x[2], midp02[0], midp12[0]]), np.array([y[2], midp02[1], midp12[1]])]
    return new_triangles

if __name__=="__main__":


    #x = np.zeros(3)
    #y = np.zeros(3)
    #for i in range(3):
        #num = np.pi/2 + 2*np.pi*i/3.
        #x[i] = np.cos(num)
        #y[i] = np.sin(num)
    #triangles = []
    #triangles.append([x,y])

    
        
    t0 = time()
    triangles = []
    for i in range(6):
        #h = np.sqrt(3)/(7*np.pi)
        h=np.sin(np.pi/3)*.5*1.5
        x = [0, np.cos(2*np.pi*i/6), np.cos(2*np.pi*(i+1)/6)]
        y = [h, h+np.sin(2*np.pi*i/6), h+np.sin(2*np.pi*(i+1)/6)]
        triangles.append([x,y])
    t1 = time()
    print 't1 ', t1-t0

    generation = 1
    params = [0.45, 0.65]
    #params = [1.25, 1.95]
    for i in xrange(generation):
        counter = 0
        new_triangles = []
        for tr in triangles:
            temp_triangles = reproduce_triangle(tr)
            for temp_tr in temp_triangles:
                new_triangles.append(temp_tr)
                counter += 1
        print i*1./generation
        triangles = new_triangles
    t2 = time()
    print 't2 ', t2-t1


    plt.figure('siers normal')
    plt.clf()
    #for i in range(3**generation):
    for t in triangles:
        #t = triangles[i]
        x = np.concatenate([t[0],[t[0][0]]])
        y = np.concatenate([t[1],[t[1][0]]])
        #plt.plot([t[0][0],t[0][1],t[0][2],t[0][0]], [t[1][0],t[1][1],t[1][2],t[1][0]])
        plt.plot(x,y)

    t3 = time()
    print 't3 ', t3-t2
    plt.figure('sierps')
    plt.clf()
        
    with PyCallGraph(output=GraphvizOutput()):
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
