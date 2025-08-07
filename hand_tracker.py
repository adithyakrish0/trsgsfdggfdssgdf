import cv2
import numpy as np

class HandTracker:
    def __init__(self):
        self.hands = []
        self.prev_hands = []
        
    def track(self, frame, skin_mask, faces):
        """Track hands in the current frame"""
        # Find potential hand regions
        hand_regions = self._find_hand_regions(skin_mask, faces)
        
        # Apply circular hand model
        detected_hands = []
        for region in hand_regions:
            hand = self._apply_circular_model(region, frame)
            if hand:
                detected_hands.append(hand)
        
        # Track hands across frames
        tracked_hands = self._track_hands(detected_hands)
        
        self.prev_hands = self.hands
        self.hands = tracked_hands
        
        return tracked_hands
    
    def _find_hand_regions(self, skin_mask, faces):
        """Find potential hand regions excluding faces"""
        # Create a mask excluding face regions
        hand_mask = skin_mask.copy()
        
        for face in faces:
            # Exclude face region from hand mask
            x, y, w, h = face.bounding_box
            cv2.rectangle(hand_mask, (x, y), (x+w, y+h), 0, -1)
        
        # Find contours in hand_mask
        contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        hand_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Minimum area threshold
                hand_regions.append(cnt)
        
        return hand_regions
    
    def _apply_circular_model(self, region, frame):
        """Apply circular hand model to region"""
        # Calculate moments
        M = cv2.moments(region)
        if M["m00"] == 0:
            return None
        
        # Calculate center
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Estimate radius
        area = cv2.contourArea(region)
        radius = int(np.sqrt(area / np.pi))
        
        # Calculate sharpness (high variation regions)
        sharpness = self._calculate_sharpness(frame, (cx, cy), radius)
        
        # Determine finger orientation using edge histogram
        orientation = self._calculate_finger_orientation(frame, (cx, cy), radius)
        
        return {
            'center': (cx, cy),
            'radius': radius,
            'sharpness': sharpness,
            'orientation': orientation,
            'confidence': sharpness * 0.7 + 0.3  # Combine metrics
        }
    
    def _calculate_sharpness(self, frame, center, radius):
        """Calculate sharpness metric for hand region"""
        x, y = center
        roi = frame[max(0, y-radius):min(frame.shape[0], y+radius), 
                    max(0, x-radius):min(frame.shape[1], x+radius)]
        
        if roi.size == 0:
            return 0.0
        
        # Apply Sum-Modulus-Difference
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharpness = np.sum(np.abs(np.diff(gray, axis=0))) + np.sum(np.abs(np.diff(gray, axis=1)))
        
        # Normalize
        max_possible = 2 * 255 * roi.shape[0] * roi.shape[1]
        return sharpness / max_possible
    
    def _calculate_finger_orientation(self, frame, center, radius):
        """Calculate finger orientation using edge histogram"""
        x, y = center
        roi = frame[max(0, y-radius):min(frame.shape[0], y+radius), 
                    max(0, x-radius):min(frame.shape[1], x+radius)]
        
        if roi.size == 0:
            return 0
        
        # Edge detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate gradient orientation
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate orientation
        orientation = np.arctan2(sobely, sobelx) * 180 / np.pi
        
        # Create histogram (20 bins as in paper)
        hist, _ = np.histogram(orientation, bins=20, range=(-180, 180))
        
        # Find dominant orientation
        dominant_bin = np.argmax(hist)
        dominant_angle = -180 + dominant_bin * 18
        
        return dominant_angle
    
    def _track_hands(self, detected_hands):
        """Track hands across frames using motion vectors"""
        tracked_hands = []
        
        # Simple tracking based on distance between frames
        for hand in detected_hands:
            matched = False
            
            for prev_hand in self.prev_hands:
                # Calculate distance between centers
                dist = np.sqrt((hand['center'][0] - prev_hand['center'][0])**2 + 
                              (hand['center'][1] - prev_hand['center'][1])**2)
                
                if dist < 50:  # Threshold for matching
                    # Update hand ID and tracking info
                    hand['id'] = prev_hand.get('id', id(hand))
                    hand['velocity'] = (
                        (hand['center'][0] - prev_hand['center'][0]),
                        (hand['center'][1] - prev_hand['center'][1])
                    )
                    matched = True
                    break
            
            if not matched:
                hand['id'] = id(hand)
                hand['velocity'] = (0, 0)
            
            tracked_hands.append(hand)
        
        return tracked_hands