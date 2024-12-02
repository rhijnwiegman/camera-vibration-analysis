import cv2

cap = cv2.VideoCapture('udp://192.168.2.3:5000')
output = cv2.VideoWriter( 
        '/home/rhijn/Camera-Stream/output2.avi' , cv2.VideoWriter_fourcc(*'XVID'), 120, (640, 480)) 

n_frames = 120 * 10

while(n_frames):
    ret, frame = cap.read()
    #output.write(frame) 
    cv2.imshow('frame',frame)
    n_frames -= 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release() 
cv2.destroyAllWindows()
