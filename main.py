from flask import Flask, render_template, Response
from camera_config_AL import RGBDhandler
import pdb
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index2.html')

def gen(RGBDhandler,ftype):
    while True:
    	rs_handler = RGBDhandler(resolution, 'bgr8', resolution, 'z16', 30)
    	color_frame, depth_frame = rs_handler.test_streamline(frame_limit = 200, is_process_depth = False)
        #frame = camera.get_frame()
    	if ftype == 'color':
        	yield (b'--frame\r\n'
                	b'Content-Type: image/jpeg\r\n\r\n' + color_frame + b'\r\n\r\n')
    	if ftype == 'depth':
        	yield (b'--frame\r\n'
        			b'Content-Type: image/jpeg\r\n\r\n' + depth_frame + b'\r\n\r\n')

@app.route('/video_feed_color')
def video_feed_color():
    return Response(gen(RGBDhandler(),ftype = 'color'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed_depth')
def video_feed_depth():
    return Response(gen(RGBDhandler(),ftype ='depth'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)