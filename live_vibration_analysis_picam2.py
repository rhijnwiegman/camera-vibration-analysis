import os
import numpy as np
import cv2
from picamera2 import Picamera2
import pandas as pd
#import plotly.express as px
#import plotly.graph_objects as go
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import scipy.stats as stats

fps = 120

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}, controls={"FrameRate": fps}))
picam2.start()

cap_width  = 640
cap_height = 480
# The following code is modified code written by Sten den Hartog!
###################################################################

# Set up parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)

# Set up parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
old_frame = picam2.capture_array()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Select ROI for tracking
bbox = cv2.selectROI(old_frame, False)
cv2.destroyAllWindows()
x, y, w, h = bbox

# if you want to manually set the ROI, comment the lines above and paste the ROI below
# x, y, w, h = 0, 0, 0, 0

roi_mask = np.zeros_like(old_gray)
roi_mask[y:y+h, x:x+w] = 255

# Detect initial points in ROI
p0 = cv2.goodFeaturesToTrack(old_gray, mask=roi_mask, **feature_params)

# Check if any points were detected
if p0 is None:
    print("Unable to determine tracking features, please adjust bounding box.")
    cap.release()  # Release the video capture object
    sys.exit()  # Exit the program

# Buffer to store the displacement of points temporarily
buffer = []
interval = 1 # interval between frames to calculate frequencies

# Draw initial tracking point
highlight_color = (0, 255, 0)  # Green
circle_radius = 5              
circle_thickness = -1          # -1 to fill the circle

frame_circle = cv2.circle(old_frame, p0.astype(int).ravel(), circle_radius, highlight_color, circle_thickness)
cv2.imshow('frame',frame_circle)

# Enable interactive plot for live frequency spectrum plotting
plt.ion()
fig, ax = plt.subplots()
 
# Process video
iteration = 0
elapsed_time = 0
while True:
    frame = picam2.capture_array()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Ensure optical flow was successful
    if p1 is not None and len(p1) > 0:
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Store points' displacements
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            buffer.append({'iteration': iteration, 'time': iteration / fps, 'id': i, 'x': a, 'y': b, 'dx': a - c, 'dy': b - d})
        # Update the previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        print("Unable to determine tracking features, please adjust bounding box.")
        break  # Break out of the loop if tracking fails
    
    frame_circle = cv2.circle(frame, good_new.astype(int).ravel(), circle_radius, highlight_color, circle_thickness)
    cv2.imshow('frame',frame_circle)

    if elapsed_time >= interval:
        # find dominant frequency using fft
        df_displacements = pd.DataFrame(buffer)
        buffer.clear()
        # Compute the FFT of the y-displacement
        time = df_displacements['time'].values
        y = df_displacements['dy'].values

        # Time step
        dt = np.mean(np.diff(time))
        # Sampling frequency
        fs = 1 / dt  

        fft_result = np.fft.fft(y)
        frequencies = np.fft.fftfreq(len(y), d=dt)

        # Only keep positive frequencies
        positive_freqs = frequencies[frequencies >= 0]
        amplitude_spectrum = np.abs(fft_result[frequencies >= 0])

        # find the dominant frequency
        dominant_frequency = positive_freqs[np.argmax(amplitude_spectrum)]
        print(dominant_frequency)
        
        # plot frequency spectrum of this interval
        ax.clear()
        ax.plot(positive_freqs, amplitude_spectrum, '-b')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Frequency spectrum")
        ax.grid(True)
        
        plt.pause(0.5)
        
        elapsed_time = 0

        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    iteration += 1
    elapsed_time += 1 / fps

cap.release()
###################################################################



cv2.destroyAllWindows()
