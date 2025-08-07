# hand_tracker_opencv.py
import cv2
import numpy as np

class HandTracker:
    def __init__(self):
        self.prev_hands = []
        self.next_id = 0
        
        # Load the pre-trained hand detection model
        # Using OpenCV's DNN module with a pre-trained model
        self.net = cv2.dnn.readNetFromCaffe(
            'deploy.prototxt', 
            'hand_detection.caffemodel'
        )
        
        # If model files are not available, fall back to contour-based detection
        self.fallback_mode = True
        
    def track(self, frame, skin_mask=None, faces=None):
        """
        Track hands in the current frame using OpenCV.
        
        Args:
            frame: Input image (BGR format).
            skin_mask: Optional skin mask for better detection.
            faces: Unused parameter (kept for compatibility).
            
        Returns:
            List of detected hands with their information.
        """
        detected_hands = []
        
        if not self.fallback_mode:
            # Use DNN-based detection
            detected_hands = self._detect_with_dnn(frame)
        else:
            # Use contour-based detection
            detected_hands = self._detect_with_contours(frame, skin_mask)
        
        # Track hands across frames
        tracked_hands = self._track_hands(detected_hands)
        
        self.prev_hands = tracked_hands
        return tracked_hands
    
    def _detect_with_dnn(self, frame):
        """
        Detect hands using OpenCV's DNN module.
        
        Args:
            frame: Input frame.
            
        Returns:
            List of detected hands.
        """
        h, w = frame.shape[:2]
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        
        # Pass blob through the network
        self.net.setInput(blob)
        detections = self.net.forward()
        
        detected_hands = []
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:  # Confidence threshold
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding box is within frame dimensions
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w - 1, endX)
                endY = min(h - 1, endY)
                
                # Extract hand ROI
                hand_roi = frame[startY:endY, startX:endX]
                if hand_roi.size == 0:
                    continue
                
                # Calculate hand center and radius
                cx = (startX + endX) // 2
                cy = (startY + endY) // 2
                radius = min(endX - startX, endY - startY) // 2
                
                # Determine if hand is open or closed
                is_open = self._is_hand_open_contour(hand_roi)
                
                detected_hands.append({
                    'id': None,
                    'center': (cx, cy),
                    'radius': radius,
                    'bounding_box': (startX, startY, endX - startX, endY - startY),
                    'orientation': 0,  # Not calculated in this method
                    'is_open': is_open,
                    'confidence': float(confidence)
                })
        
        return detected_hands
    
    def _detect_with_contours(self, frame, skin_mask):
        """
        Detect hands using contour-based approach.
        
        Args:
            frame: Input frame.
            skin_mask: Optional skin mask for better detection.
            
        Returns:
            List of detected hands.
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Extract skin mask
        if skin_mask is None:
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_hands = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < 1000:  # Minimum area threshold
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = float(w) / h
            
            # Filter by aspect ratio (hands are typically not too wide or too tall)
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # Extract hand ROI
            hand_roi = frame[y:y+h, x:x+w]
            if hand_roi.size == 0:
                continue
            
            # Calculate hand center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Calculate radius
            radius = int(np.sqrt(area / np.pi))
            
            # Determine if hand is open or closed
            is_open = self._is_hand_open_contour(hand_roi)
            
            detected_hands.append({
                'id': None,
                'center': (cx, cy),
                'radius': radius,
                'bounding_box': (x, y, w, h),
                'orientation': 0,  # Not calculated in this method
                'is_open': is_open,
                'confidence': min(1.0, area / 10000)  # Simple confidence based on area
            })
        
        return detected_hands
    
    def _is_hand_open_contour(self, hand_roi):
        """
        Determine if hand is open or closed using contour analysis.
        
        Args:
            hand_roi: Hand region of interest.
            
        Returns:
            Boolean indicating if hand is open.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        # Find the largest contour (hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate convex hull
        hull = cv2.convexHull(largest_contour)
        
        # Calculate convexity defects
        try:
            hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
            defects = cv2.convexityDefects(largest_contour, hull_indices)
            
            if defects is None:
                return False
            
            # Count significant defects (potential gaps between fingers)
            significant_defects = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                if d > 5000:  # Threshold for significant defect
                    significant_defects += 1
            
            # Consider hand open if there are at least 3 significant defects
            return significant_defects >= 3
        except:
            # Fallback: use contour area vs. convex hull area ratio
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(largest_contour)
            
            if hull_area == 0:
                return False
            
            solidity = float(contour_area) / hull_area
            return solidity < 0.85  # Open hands typically have lower solidity
    
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
    
    def draw_debug_info(self, frame, hands):
        """
        Draw debug information on the frame.
        
        Args:
            frame: Input frame.
            hands: List of detected hands.
            
        Returns:
            Frame with debug information drawn.
        """
        for hand in hands:
            cx, cy = hand['center']
            radius = hand['radius']
            x, y, w, h = hand['bounding_box']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw hand circle
            cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
            
            # Draw hand ID
            cv2.putText(frame, f"Hand {hand['id']}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw hand state (open/closed)
            state = "Open" if hand['is_open'] else "Closed"
            cv2.putText(frame, state, (x, y + h + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw confidence
            cv2.putText(frame, f"{hand['confidence']:.2f}", (x + w - 50, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame