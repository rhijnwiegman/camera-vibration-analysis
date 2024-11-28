import cv2
import subprocess

cap = cv2.VideoCapture('udp://192.168.2.3:5000')
output = cv2.VideoWriter( 
        '/home/rhijn/Camera-Stream/output.avi' , cv2.VideoWriter_fourcc(*'XVID'), 120, (640, 480)) 

n_frames = 120 * 10

while(n_frames):
    ret, frame = cap.read()
    output.write(frame) 
    cv2.imshow('frame',frame)
    n_frames -= 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release() 
cv2.destroyAllWindows()

###############################################

shell_script = '/home/rhijn/STB-VMM-master/magnify_video.sh'

# Usage: 
#   
#   The following arguments must be provided:
#   -mag (magnification factor): Video magnification factor (default 20)
#   -i (input file): Path pointing to target video (required)
#   -s (save dir): Path to a directory to store result files (required)
#   -m (model checkpoint): Path to the last model checkpoint (required)
#   -o (output): Output project name (required)
#   -mod (mode): static(default)/dynamic (params default)
#   -f (framerate): Framerate of the input video (default 60)
#   -c (cuda): Activates cuda (default cpu)

arguments = [
    '-mag', '20',
    '-i', '/home/rhijn/Camera-Stream/output.avi',
    '-s', '/home/rhijn/Camera-Stream/magnified_videos',
    '-m', '/home/rhijn/STB-VMM-master/ckpt/ckpt_e49.pth.tar',
    '-o', 'magnified_video_stream_stb-vmm',
    '-f', '120',
    '-c' 
]

command = ['bash', shell_script] + arguments

try:
    subprocess.run(command, check=True, cwd='/home/rhijn/STB-VMM-master/')
    print('applied stb-vmm to video successfully')
except subprocess.CalledProcessError as e:
    print(f'error while running the script {e}')