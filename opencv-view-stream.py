import cv2
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}, controls={"FrameRate": 120}))
picam2.start()
#output = cv2.VideoWriter( '/home/rhijn/Camera-Stream/output2.avi' , cv2.VideoWriter_fourcc(*'XVID'), 120, (640, 480)) 

n_frames = 120 * 10

while(n_frames):
    frame = picam2.capture_array()
    #output.write(frame) 
    cv2.imshow('frame',frame)
    n_frames -= 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release() 
cv2.destroyAllWindows()
