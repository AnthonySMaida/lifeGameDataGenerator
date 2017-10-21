# -*- coding: utf-8 -*-
"""
Author: Maida
Created: July 23, 2017
Modified: Oct 21, 2017 to create a video file

Creates a video of the Game of Life and converts each frame
to a 4D tensor to use with TensorFlow.

The Python frames are printed using matplotlib.
View the tensor frames in TensorBoard as follows:
    tensorboard --logdir /tmp/lifevideo/1/
    In browser, open http address given.
    Then click "IMAGES" tab.
    Then click "videoFrames"
    
If ffmpeg is not installed, comment out line: 113
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

import numpy as np
import random
import tensorflow as tf

def life_step_2(X):  # this function was given to me by a friend
    """Game of life step using scipy tools"""
    from scipy.signal import convolve2d
#    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X # original
    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='fill') - X
    return (nbrs_count == 3) | (X & (nbrs_count == 2))

LOGDIR = "/tmp/lifeVideo/"    # for tensorboard
LENGTH_OF_VID = 16
BOARD_SZ      =  9   # 9x9

# Store video in ndarray with three dimensions
video = np.empty([LENGTH_OF_VID, BOARD_SZ, BOARD_SZ], dtype=np.int8)

# Create an initial random board
for i in range(BOARD_SZ):
    for j in range(BOARD_SZ):
        video[0,i,j] = random.randint(0,1) # init board to random 0s & 1s
        
for i in range(LENGTH_OF_VID - 1):
    video[i+1,:,:] = life_step_2(video[i,:,:]) # simulate game for n - 1 steps

print("Video frames :", video.shape)
for i in range(LENGTH_OF_VID):
    print("Frame: ", i)
    plt.gca().invert_yaxis()
    plt.imshow(video[i,:,:], cmap = cm.Greys_r) # print the board states
    plt.show()
    
raw_frame = video[0, :, :]
raw_frame = np.expand_dims(raw_frame, axis = -1)
print("raw_frame.shape: ", raw_frame.shape)

graph = tf.Graph()
with graph.as_default():
    
    # cvrt video to tensors
    tensor_frames = []
    for i in range(LENGTH_OF_VID): # create a list of tensor frames
        raw_frame = video[i, :, :]
        tf_frame = tf.constant(raw_frame, dtype=tf.float32)
        tf_frame = tf.expand_dims(tf_frame, axis = 0)
        tf_frame = tf.expand_dims(tf_frame, axis = -1) # make 4D
        tensor_frames.append(tf_frame)
        
    with tf.name_scope("videoFrames"): # make hook to summarize for tensorboard
        for i in range(LENGTH_OF_VID):
            tf.summary.image("frame"+str(i), tensor_frames[i], 1)

    
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    
    # Create graph summary
    # Use a different log file each time you run the program.
    msumm = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR + "1") # += 1 for each run till /tmp is cleard
    writer.add_graph(sess.graph)
    
    evaled_frames = []
    for i in range(LENGTH_OF_VID):
        evaled_frames.append(tensor_frames[i].eval())
        print("shaped eval'd frame: ", evaled_frames[i].shape)
        ms = sess.run(msumm) # merge summary
        writer.add_summary(ms, i)
        
"""
How to write an animiation file.
Does not need the TensorFlow results (based on numpy video).
Writes file 'life_anim.mp4' to current directory using save().

File writing needs ffmpeg to be installed.
If using Anaconda Python, do this by typing:
    conda install -c conda-forge ffmpeg
at the terminal.
"""
fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.axis('equal')
images = []
for frame in range(LENGTH_OF_VID):
    images.append((plt.pcolormesh(video[frame,:,:], cmap = cm.Greys_r),))
    
life_anim = animation.ArtistAnimation(fig1, images, interval=100, repeat_delay=3000, blit=True)

# Comment out the next line if ffmpet is not installed
life_anim.save('life_anim.mp4', metadata={'artist':'Tony'}) # This writes the full animation

# This will only show the last frame in the animation.
plt.show()
