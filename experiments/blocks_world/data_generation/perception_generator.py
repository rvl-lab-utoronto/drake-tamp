
import pyrealsense2 as rs
import numpy as np
import cv2
from pupil_apriltags import Detector 

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

#apriltag setup 
detector = Detector(families='tagStandard41h12')
#rs-enumerate-devices -c 
#https://github.com/IntelRealSense/librealsense/issues/3555

# fx, fy = 1363.168457, 1361.44909668
# cx, cy = 638.760315, 341.2155761

fx, fy = 1363.168457, 1361.44909668
cx, cy = 958.1405029, 511.8233643
intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

#get pose in camera space 
def get_pixel(pose):
    pixel_space = np.matmul(intrinsic_matrix, pose)
    return (int(pixel_space[0]/pixel_space[2]), int(pixel_space[1]/pixel_space[2]))


#get vector from cube frame of reference and return pose in camera frame
def get_camera_pos(vec, R, t):
    return t + np.matmul(R, vec)

cw = 0.05
bound_cube_cords = [
    [[cw/2], [cw/2], [0]],
    [[-cw/2], [cw/2], [0]],
    [[cw/2], [-cw/2], [0]],
    [[-cw/2], [-cw/2], [0]],
    [[cw/2], [cw/2], [cw]],
    [[-cw/2], [cw/2], [cw]],
    [[cw/2], [-cw/2], [cw]],
    [[-cw/2], [-cw/2], [cw]]
]

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        results = detector.detect(gray, estimate_tag_pose=True, camera_params=[fx, fy, cx, cy], tag_size=(6/10)*0.05)
        image = color_image.copy()
        
        # Show images
        for r in results:
            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            
            # draw the bounding box of the AprilTag detection
            cv2.line(image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(image, ptD, ptA, (0, 255, 0), 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the image
            tagFamily = r.tag_family.decode("utf-8")
            cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #print("[INFO] tag family: {}".format(tagFamily))

            t = r.pose_t
            R = r.pose_R
            #print("error", r.pose_err)
            print("pose", t)

            #draw cube 
            for vert1 in bound_cube_cords:
                p1 = get_pixel(get_camera_pos(vert1, R, t))
                cv2.circle(image, p1, 5, (255, 0, 0), -1)    
                for vert2 in bound_cube_cords: 
                    p2 = get_pixel(get_camera_pos(vert2, R, t))
                    if np.linalg.norm(np.subtract(vert1, vert2)) < cw + 0.01:
                        cv2.line(image, p1, p2, (0, 0, 255), 1)    
            
            #draw triad 
            x_line = t + np.matmul(R, [[0.1], [0], [0]])
            y_line = t + np.matmul(R, [[0], [0.1], [0]])
            z_line = t + np.matmul(R, [[0], [0], [0.1]])
            cv2.circle(image, get_pixel(t), 5, (255, 0, 0), -1)
            cv2.line(image, get_pixel(t), get_pixel(x_line), (0, 0, 255), 2)
            cv2.line(image, get_pixel(t), get_pixel(y_line), (0, 255, 0), 2)
            cv2.line(image, get_pixel(t), get_pixel(z_line), (255, 0, 0), 2)

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()