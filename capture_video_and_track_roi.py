import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

from tqdm.notebook import tqdm
from tqdm import tqdm

# capture video of 10s
# cap = cv2.VideoCapture('udp://192.168.2.3:5000')
# output = cv2.VideoWriter( 
#         '/home/rhijn/Camera-Stream/output.avi' , cv2.VideoWriter_fourcc(*'XVID'), 120, (640, 480)) 

# n_frames = 120 * 10

# while(n_frames):
#     ret, frame = cap.read()
#     output.write(frame) 
#     cv2.imshow('frame',frame)
#     n_frames -= 1
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# output.release() 
# cv2.destroyAllWindows()

fps = 120

cap = cv2.VideoCapture('videos/koelkast_videos_640_480_120fps_24cm_6mmlens\koel_take1.h264') # vervang dit path naar het path van je eigen video
cap_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# The following code is modified code written by Jonas Schoonhoven!
###################################################################

# Set up parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)

# Set up parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
ret, old_frame = cap.read()
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

# List to store the displacement of points
displacements = []

output = cv2.VideoWriter( 
        'C:/Users/Gebruiker/Videos/roi_tracked_output.avi' , cv2.VideoWriter_fourcc(*'XVID'), fps, (cap_width, cap_height))

# Draw initial tracking point
highlight_color = (0, 255, 0)  # Green
circle_radius = 5              
circle_thickness = -1          # -1 to fill the circle

frame_circle = cv2.circle(old_frame, p0.astype(int).ravel(), circle_radius, highlight_color, circle_thickness)
output.write(frame_circle)

# Process video
iteration = 0
with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Tracking") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
                displacements.append({'iteration': iteration, 'time': pbar.n / fps, 'id': i, 'x': a, 'y': b, 'dx': a - c, 'dy': b - d})

            # Update the previous frame and points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            print("Unable to determine tracking features, please adjust bounding box.")
            break  # Break out of the loop if tracking fails
        
        frame_circle = cv2.circle(frame, good_new.astype(int).ravel(), circle_radius, highlight_color, circle_thickness)
        output.write(frame_circle)

        pbar.update(1)
        iteration += 1

cap.release()
output.release()
###################################################################

# Convert the list of displacements to a DataFrame
df_displacements = pd.DataFrame(displacements)

# Plot the trajectory of the ROI 
fig = px.scatter(df_displacements, x='x', y='y', text='time', title='Trajectory of ROI')
fig.add_scatter(x=df_displacements['x'], y=df_displacements['y'],
                mode='lines',
                line=dict(color='blue', width=2))
fig.update_layout(
    xaxis_title='X position',
    yaxis_title='Y position',
    showlegend=True
)
fig.show()

# Plot the oscillation of the ROI in the y-direction
fig = px.line(df_displacements, x='time', y='y', title='Oscillation of ROI in y-direction')
fig.update_layout(
    xaxis_title='Time (s)',
    yaxis_title='Y position'
)
fig.show()

# Plot the rate of change of the ROI in the y-direction (dy)
fig = px.line(df_displacements, x='time', y='dy', title='Rate of change in y-direction (dy)')
fig.update_layout(
    xaxis_title='Time (s)',
    yaxis_title='DY'
)
fig.show()

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

fig = px.line(x=positive_freqs, y=amplitude_spectrum, title='Frequency Spectrum')
fig.update_layout(
    xaxis_title='Frequency (Hz)',
    yaxis_title='Amplitude'
)
fig.show()
print("dominant frequency:", np.argmax(amplitude_spectrum))