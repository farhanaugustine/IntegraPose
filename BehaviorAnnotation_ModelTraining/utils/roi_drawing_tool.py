#roi_drawing_tool.py
import cv2

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    frame = param['frame']
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        if len(points) > 1:
            cv2.line(frame, points[-2], points[-1], (0, 255, 0), 2)
        cv2.imshow("Draw ROI", frame)

def draw_roi(frame):
    """
    Displays a frame and allows the user to draw a polygon ROI.
    Returns the list of polygon vertices.
    """
    global points
    points = []
    
    clone = frame.copy()
    cv2.namedWindow("Draw ROI")
    cv2.setMouseCallback("Draw ROI", mouse_callback, {'frame': clone})

    while True:
        cv2.imshow("Draw ROI", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'): # Reset
            clone = frame.copy()
            points = []
        elif key == 13: # Enter key to finish
            break
        elif key == 27: # Escape key to cancel
            points = []
            break
            
    cv2.destroyWindow("Draw ROI")
    return points