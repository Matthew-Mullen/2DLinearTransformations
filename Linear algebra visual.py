#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# x values span from -4 to 4 with 9 evenly spaced samples
# y values span from -3 to 3 with 7 evenly spaced samples
xvals = np.linspace(-4, 4, 9)
yvals = np.linspace(-3, 3, 7)
# this is our original grid before any linear transformation
xygrid = np.column_stack([[x, y] for x in xvals for y in yvals])
# Create matrix to perform linear transformation
a = np.column_stack([[2, 1], [-1, 1]])
#print(a)
# create transformed grid created by taking the dot product of our 2x2 matrix a and our xygrid
uvgrid = np.dot(a, xygrid)


# This function assigns a unique color based on position
def colorizer(x, y):
    """
    Map x-y coordinates to a rgb color in order to show user where points are going
    """
    
    r = min(1, 1-y/3)
    g = min(1, 1+y/3)
    b = 1/4 + x/16
    return (r, g, b)
# Map grid coordinates to colors and contains a list of the colors
# items are sent to the functio colorizer as parameters and iterated over our map then returns a list of colors for each point
colors = list(map(colorizer, xygrid[0], xygrid[1]))




# Plot grid points before transformation
plt.figure(figsize=(6, 6), facecolor="w")
plt.scatter(xygrid[0], xygrid[1], s=36, c=colors, edgecolor="black")
# Set axis limits
plt.grid(True)
plt.axis("equal")
# meaning space spanned by the values we set of x and v
plt.title("Pre-transformation grid in x-y space")


# Plot transformed grid points
plt.figure(figsize=(6, 6), facecolor="w")
plt.scatter(uvgrid[0], uvgrid[1], s=36, c=colors, edgecolor="black")
plt.grid(True)
plt.axis("equal")
plt.title("Post-Transformation grid in u-v space")

# Now to generate an animation we need to come up with intermediate transforms so that the program has something to animate
def transformation_steps(matrix,arrayofcoords,numsteps=100):
    # to achieve this is by constructing a series of 2-by-2 matrices that interpolate between the identity matrix
    # and the target matrix (in the above case a)
    '''
    Generate a series of intermediate transform for the matrix multiplication
      np.dot(matrix, arrayofcoords) # matrix multiplication
    starting with the identity matrix, where
      matrix: 2-by-2 matrix
      arrayofcoords: 2-by-n array of coordinates in x-y space 

    Returns a (numsteps + 1)-by-2-by-n array (3d array)
    '''
    # creates empty grid of size we want. np.shape gets us the tuple of dimension of arrayofcoords
    transgrid=np.zeros((numsteps+1,)+np.shape(arrayofcoords))
    # computing intermediate transforms
    for j in range(numsteps+1):
        # np.eye returns a 2d array in RREF form (ones on the diagonal) 
        intermediate = np.eye(2)+j/numsteps*(matrix-np.eye(2))
        transgrid[j]=np.dot(intermediate,arrayofcoords)
    return transgrid



transform=transformation_steps(a,xygrid,30)

# To construct the animated version, we need to save each of these intermediate plots as an image file. 
# The following code block defines a function that generates a series of image files with the filename 
# frame-xx.png and saves them in a subdirectory. We apply this function to the array of intermediate grid 
# coordinates that we generated above:
# Create a series of figures showing the intermediate transforms
def make_plots(transarray, color, outdir="png-frames", figuresize=(4,4), figuredpi=150):
    '''
    Generate a series of png images showing a linear transformation stepwise
    '''
    nsteps = transarray.shape[0]
    ndigits = len(str(nsteps)) # to determine filename padding
    maxval = np.abs(transarray.max()) # to set axis limits
    # create directory if necessary
    import os
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # create figure
    plt.ioff()
    fig = plt.figure(figsize=figuresize, facecolor="w")
    for j in range(nsteps): # plot individual frames
        plt.cla()
        plt.scatter(transarray[j,0], transarray[j,1], s=36, c=color, edgecolor="none")
        plt.xlim(1.1*np.array([-maxval, maxval]))
        plt.ylim(1.1*np.array([-maxval, maxval]))
        plt.grid(True)
        plt.draw()
        # save as png
        outfile = os.path.join(outdir, "frame-" + str(j+1).zfill(ndigits) + ".png")
        fig.savefig(outfile, dpi=figuredpi)
    plt.ion()

# Generate figures
make_plots(transform, colors, outdir="tmp")

# Convert to gif (works on linux/os-x, requires image-magick)
from subprocess import call
call("cd png-frames && convert -delay 10 frame-*.png ../animation.gif", shell=True)
# Optional: uncomment below clean up png files
#call("rm -f png-frames/*.png", shell=True)


# In[ ]:


# Shear matrix transformation
# A shear matrix of the form stretches the grid along the x axis by an amount proportional to the y coordinate of a point.
a = np.column_stack([[1, 0], [2, 1]]) # shear along x-axis
print(a)
# Generate intermediates
transform = transformation_steps(a, xygrid, 30)
make_plots(transform, colors)
# above creates the animation


# In[ ]:


# Permutation
# this interchanges rows and cols
a = np.column_stack([[0, 1], [1, 0]])
print(a)
# Generate intermediates
transform = transformation_steps(a, xygrid, 30)
make_plots(transform, colors)
# above creates the animation

