# hand_tracker.py
import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, static_image_mode=False, max_num_hands=2, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the MediaPipe Hands solution.
        
        Args:
            static_image_mode: Whether to treat the input images as a batch of static
                and possibly unrelated images, or as a video stream.
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence value for hand detection.
            min_tracking_confidence: Minimum confidence value for hand tracking.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.prev_hands = []
        self.next_id = 0
        
    def track(self, frame, skin_mask=None, faces=None):
        """
        Track hands in the current frame using MediaPipe Hands.
        
        Args:
            frame: Input image (BGR format).
            skin_mask: Unused parameter (kept for compatibility).
            faces: Unused parameter (kept for compatibility).
            
        Returns:
            List of detected hands with their information.
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get hand landmarks
        results = self.hands.process(rgb_frame)
        
        detected_hands = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand information
                hand_info = self._extract_hand_info(hand_landmarks, frame, idx)
                if hand_info:
                    detected_hands.append(hand_info)
        
        # Track hands across frames
        tracked_hands = self._track_hands(detected_hands)
        
        self.prev_hands = tracked_hands
        return tracked_hands
    
    def _extract_hand_info(self, hand_landmarks, frame, idx):
        """
        Extract hand information from landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks.
            frame: Input frame for size reference.
            idx: Index of the hand in the detection results.
            
        Returns:
            Dictionary containing hand information.
        """
        # Get image dimensions
        h, w, _ = frame.shape
        
        # Calculate hand center (using wrist landmark)
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        cx, cy = int(wrist.x * w), int(wrist.y * h)
        
        # Calculate hand radius (distance from wrist to middle finger MCP)
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        mx, my = int(middle_mcp.x * w), int(middle_mcp.y * h)
        radius = int(np.sqrt((cx - mx)**2 + (cy - my)**2))
        
        # Calculate bounding box
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Calculate hand orientation (angle between wrist and middle finger)
        orientation = np.arctan2(my - cy, mx - cx) * 180 / np.pi
        
        # Determine if hand is open or closed
        is_open = self._is_hand_open(hand_landmarks, h, w)
        
        # Calculate confidence (average of landmark visibility)
        confidence = sum([landmark.visibility for landmark in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
        
        return {
            'id': None,  # Will be assigned in tracking
            'center': (cx, cy),
            'radius': radius,
            'bounding_box': (x_min, y_min, x_max - x_min, y_max - y_min),
            'orientation': orientation,
            'is_open': is_open,
            'confidence': confidence,
            'landmarks': hand_landmarks
        }
    
    def _is_hand_open(self, hand_landmarks, h, w):
        """
        Determine if the hand is open or closed based on finger positions.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks.
            h, w: Frame height and width.
            
        Returns:
            Boolean indicating if the hand is open.
        """
        # Get key landmarks
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        
        # Convert to pixel coordinates
        def to_pixel(landmark):
            return int(landmark.x * w), int(landmark.y * h)
        
        wrist_px = to_pixel(wrist)
        thumb_tip_px = to_pixel(thumb_tip)
        thumb_ip_px = to_pixel(thumb_ip)
        index_tip_px = to_pixel(index_tip)
        index_pip_px = to_pixel(index_pip)
        middle_tip_px = to_pixel(middle_tip)
        middle_pip_px = to_pixel(middle_pip)
        ring_tip_px = to_pixel(ring_tip)
        ring_pip_px = to_pixel(ring_pip)
        pinky_tip_px = to_pixel(pinky_tip)
        pinky_pip_px = to_pixel(pinky_pip)
        
        # Calculate distances from fingertips to PIP joints
        index_dist = np.sqrt((index_tip_px[0] - index_pip_px[0])**2 + (index_tip_px[1] - index_pip_px[1])**2)
        middle_dist = np.sqrt((middle_tip_px[0] - middle_pip_px[0])**2 + (middle_tip_px[1] - middle_pip_px[1])**2)
        ring_dist = np.sqrt((ring_tip_px[0] - ring_pip_px[0])**2 + (ring_tip_px[1] - ring_pip_px[1])**2)
        pinky_dist = np.sqrt((pinky_tip_px[0] - pinky_pip_px[0])**2 + (pinky_tip_px[1] - pinky_pip_px[1])**2)
        
        # Calculate thumb distance
        thumb_dist = np.sqrt((thumb_tip_px[0] - thumb_ip_px[0])**2 + (thumb_tip_px[1] - thumb_ip_px[1])**2)
        
        # Calculate hand size (wrist to middle finger MCP)
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        middle_mcp_px = to_pixel(middle_mcp)
        hand_size = np.sqrt((wrist_px[0] - middle_mcp_px[0])**2 + (wrist_px[1] - middle_mcp_px[1])**2)
        
        # Normalize distances by hand size
        index_dist_norm = index_dist / hand_size
        middle_dist_norm = middle_dist / hand_size
        ring_dist_norm = ring_dist / hand_size
        pinky_dist_norm = pinky_dist / hand_size
        thumb_dist_norm = thumb_dist / hand_size
        
        # Thresholds for determining if fingers are extended
        threshold = 0.4
        
        # Count extended fingers
        extended = 0
        if index_dist_norm > threshold:
            extended += 1
        if middle_dist_norm > threshold:
            extended += 1
        if ring_dist_norm > threshold:
            extended += 1
        if pinky_dist_norm > threshold:
            extended += 1
        if thumb_dist_norm > threshold * 0.8:  # Thumb has different threshold
            extended += 1
        
        # Consider hand open if at least 3 fingers are extended
        return extended >= 3
    
    def _track_hands(self, detected_hands):
        """
        Track hands across frames using simple distance matching.
        
        Args:
            detected_hands: List of newly detected hands.
            
        Returns:
            List of tracked hands with consistent IDs.
        """
        tracked_hands = []
        
        for hand in detected_hands:
            matched = False
            
            # Try to match with previous hands
            for prev_hand in self.prev_hands:
                # Calculate distance between centers
                dist = np.sqrt((hand['center'][0] - prev_hand['center'][0])**2 + 
                              (hand['center'][1] - prev_hand['center'][1])**2)
                
                if dist < 50:  # Threshold for matching
                    # Update hand ID and tracking info
                    hand['id'] = prev_hand['id']
                    hand['velocity'] = (
                        (hand['center'][0] - prev_hand['center'][0]),
                        (hand['center'][1] - prev_hand['center'][1])
                    )
                    matched = True
                    break
            
            if not matched:
                hand['id'] = self.next_id
                self.next_id += 1
                hand['velocity'] = (0, 0)
            
            tracked_hands.append(hand)
        
        return tracked_hands
    
    def draw_landmarks(self, frame, hands):
        """
        Draw hand landmarks on the frame for visualization.
        
        Args:
            frame: Input frame.
            hands: List of detected hands.
            
        Returns:
            Frame with drawn landmarks.
        """
        if hands:
            for hand in hands:
                if 'landmarks' in hand:
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand['landmarks'], 
                        self.mp_hands.HAND_CONNECTIONS
                    )
        return frame