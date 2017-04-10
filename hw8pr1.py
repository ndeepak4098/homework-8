#
# coding: utf-8
#
# hw8pr1.py - the k-means algorithm -- with pixels...
#

# import everything we need...
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import utils
import cv2

# choose an image...
#IMAGE_NAME = "./jp.png"  # Jurassic Park
#IMAGE_NAME = "./batman.png"
#IMAGE_NAME = "./hmc.png"
#IMAGE_NAME = "./thematrix.png"
#IMAGE_NAME = "./fox.jpg"
IMAGE_NAME = "./gr.jpg"
image = cv2.imread(IMAGE_NAME, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# reshape the image to be a list of pixels
image_pixels = image.reshape((image.shape[0] * image.shape[1], 3))

# choose k (the number of means) in  NUM_MEANS
# and cluster the pixel intensities

NUM_MEANS = 4
clusters = KMeans(n_clusters = NUM_MEANS)
clusters.fit(image_pixels)

# After the call to fit, the key information is contained
# in  clusters.cluster_centers_ :
count = 0
for center in clusters.cluster_centers_:
    print("Center #", count, " == ", center)
    # note that the center's values are floats, not ints!
    center_integers = [int(p) for p in center]
    print("   and as ints:", center_integers)
    count += 1

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clusters)
bar = utils.plot_colors(hist, clusters.cluster_centers_)


# in the first figure window, show our image
plt.figure()
plt.axis("off")
plt.imshow(image)

# in the second figure window, show the pixel histograms 
#   this starter code has a single value of k for each
#   your task is to vary k and show the resulting histograms
# this also illustrates one way to display multiple images
# in a 2d layout (fig == figure, ax == axes)
#
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
title = str(NUM_MEANS)+" means"
ax[0,0].imshow(bar);    ax[0,0].set_title(title)
ax[0,1].imshow(bar);    ax[0,1].set_title(title)
ax[1,0].imshow(bar);    ax[1,0].set_title(title)
ax[1,1].imshow(bar);    ax[1,1].set_title(title)
for row in range(2):
    for col in range(2):
        ax[row,col].axis('off')
plt.show(fig)

def replace_images(im, kmeans):

    """  pixels into a list call kmeans"""

    image = im.copy()

    # reshape the image to be a list of pixels
    image_pixels = image.reshape((image.shape[0] * image.shape[1], 3))

    # choose k (the number of means) in  NUM_MEANS
    # and cluster the pixel intensities

    NUM_MEANS = kmeans
    clusters = KMeans(n_clusters = NUM_MEANS)
    clusters.fit(image_pixels)

    # After the call to fit, the key information is contained
    # in  clusters.cluster_centers_ :
    count = 0
    new_list = []
    for center in clusters.cluster_centers_:
        print("Center #", count, " == ", center)
        # note that the center's values are floats, not ints!
        center_integers = [int(p) for p in center]
        print("   and as ints:", center_integers)
        new_list += [center_integers]
        count += 1
    print(new_list)
    

    num_rows, num_cols, num_chans = image.shape
    
    for row in range(num_rows):
        for col in range(num_cols):
                pixel = image[row,col]
                pr,pg,pb = pixel

                distances = []

                for i in new_list:
                    distance1 = abs(i[0] - pr) + abs(i[1] - pg) + abs(i[2] - pb)
                    distances += [distance1]
                        

                #distances = []

                #distance1 = abs(new_list[0][0] - pr) + abs(new_list[0][1] - pg) + abs(new_list[0][2] - pb)
                #distances += [distance1]
                #distance2 = abs(new_list[1][0] - pr) + abs(new_list[1][1] - pg) + abs(new_list[1][2] - pb)
                #distances += [distance2]
                #distance3 = abs(new_list[2][0] - pr) + abs(new_list[2][1] - pg) + abs(new_list[2][2] - pb)
                #distances += [distance3]
                #distance4 = abs(new_list[3][0] - pr) + abs(new_list[3][1] - pg) + abs(new_list[3][2] - pb)
                #distances += [distance4]

                if min(distances) == distances[0]:
                    image[row,col] = new_list[0]
                elif min(distances) == distances[1]:
                    image[row,col] = new_list[1]
                elif min(distances) == distances[2]:
                    image[row,col] = new_list[2]
                else:
                    image[row,col] = new_list[3]
    return image
                
            
from matplotlib import pyplot as plt
# Reading and color-converting an image to RGB
raw_image = cv2.imread("./gr.jpg",cv2.IMREAD_COLOR) 
#raw_image = cv2.imread('monalisa.jpg',cv2.IMREAD_COLOR) 

# convert an OpenCV image (BGR) to an "ordinary" image (RGB) 
image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

new_image = replace_images(image, 7)
plt.imshow(new_image)
plt.show()


#
# comments and reflections on hw8pr1, k-means and pixels
"""
 + Which of the paths did you take:  
    + posterizing or 
    + algorithm-implementation

 + How did it go?  Which file(s) should we look at?
 + Which function(s) should we try...
"""
#
#