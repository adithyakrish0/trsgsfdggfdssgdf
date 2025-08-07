import cv2
import numpy as np

class BottleTracker:
    def __init__(self):
        self.bottles = []
        self.prev_bottles = []
        self.bottle_templates = self._initialize_bottle_templates()
        
    def _initialize_bottle_templates(self):
        """Initialize bottle template library"""
        # In a real implementation, this would be loaded from a database
        # of common medication bottles
        return []
    
    def track(self, frame, skin_mask, hands):
        """Track medication bottles in the current frame"""
        # Find potential bottle regions
        bottle_regions = self._find_bottle_regions(frame)
        
        # Match against template library
        detected_bottles = []
        for region in bottle_regions:
            bottle = self._match_bottle_template(region, frame)
            if bottle:
                detected_bottles.append(bottle)
        
        # Track bottles across frames
        tracked_bottles = self._track_bottles(detected_bottles, hands)
        
        self.prev_bottles = self.bottles
        self.bottles = tracked_bottles
        
        return tracked_bottles
    
    def _find_bottle_regions(self, frame):
        """Find potential bottle regions using edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by shape (rectangular with aspect ratio ~2:1)
        bottle_regions = []
        for cnt in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Check aspect ratio (bottles are typically taller than wide)
            aspect_ratio = h / w
            if 1.5 < aspect_ratio < 3.0:
                # Check area
                area = cv2.contourArea(cnt)
                if area > 1000:  # Minimum area threshold
                    bottle_regions.append({
                        'contour': cnt,
                        'bounding_box': (x, y, w, h),
                        'aspect_ratio': aspect_ratio,
                        'area': area
                    })
        
        return bottle_regions
    
    def _match_bottle_template(self, region, frame):
        """Match bottle region against template library"""
        x, y, w, h = region['bounding_box']
        
        # Extract ROI
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return None
        
        best_match = None
        best_score = 0
        
        # Match against each template in the library
        for template in self.bottle_templates:
            # Resize template to match ROI size
            resized_template = cv2.resize(template['image'], (w, h))
            
            # Calculate similarity score (SSD)
            score = self._calculate_ssd(roi, resized_template)
            
            if score > best_score:
                best_score = score
                best_match = template
        
        if best_score > 0.7:  # Threshold for positive match
            return {
                'template': best_match,
                'position': (x, y),
                'size': (w, h),
                'confidence': best_score,
                'state': 'unknown'  # open, closed, unknown
            }
        
        return None
    
    def _calculate_ssd(self, roi, template):
        """Calculate Sum of Squared Differences between ROI and template"""
        if roi.shape != template.shape:
            return 0.0
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
            
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template
        
        # Calculate SSD
        diff = roi_gray.astype(np.float32) - template_gray.astype(np.float32)
        ssd = np.sum(diff ** 2)
        
        # Normalize to [0, 1]
        max_possible = 255 * 255 * roi_gray.size
        similarity = 1.0 - (ssd / max_possible)
        
        return similarity
    
    def _track_bottles(self, detected_bottles, hands):
        """Track bottles across frames and handle occlusions"""
        tracked_bottles = []
        
        for bottle in detected_bottles:
            matched = False
            
            # Try to match with previous bottles
            for prev_bottle in self.prev_bottles:
                # Calculate distance between centers
                dist = np.sqrt((bottle['position'][0] - prev_bottle['position'][0])**2 + 
                              (bottle['position'][1] - prev_bottle['position'][1])**2)
                
                if dist < 50:  # Threshold for matching
                    # Update bottle ID and tracking info
                    bottle['id'] = prev_bottle.get('id', id(bottle))
                    bottle['velocity'] = (
                        (bottle['position'][0] - prev_bottle['position'][0]),
                        (bottle['position'][1] - prev_bottle['position'][1])
                    )
                    
                    # Check for state change (opening/closing)
                    bottle['state'] = self._detect_bottle_state(bottle, hands)
                    bottle['state_changed'] = (bottle['state'] != prev_bottle.get('state', 'unknown'))
                    
                    matched = True
                    break
            
            if not matched:
                bottle['id'] = id(bottle)
                bottle['velocity'] = (0, 0)
                bottle['state'] = self._detect_bottle_state(bottle, hands)
                bottle['state_changed'] = False
            
            tracked_bottles.append(bottle)
        
        # Handle occlusions
        tracked_bottles = self._handle_occlusions(tracked_bottles, hands)
        
        return tracked_bottles
    
    def _detect_bottle_state(self, bottle, hands):
        """Detect if bottle is open or closed based on hand positions"""
        # Check if hands are near the bottle cap
        x, y = bottle['position']
        w, h = bottle['size']
        
        # Cap region (top 20% of bottle)
        cap_top = y
        cap_bottom = y + int(0.2 * h)
        cap_left = x
        cap_right = x + w
        
        # Check if any hand is in the cap region
        for hand in hands:
            hx, hy = hand['center']
            hr = hand['radius']
            
            # Check if hand overlaps with cap region
            if (cap_left - hr < hx < cap_right + hr and 
                cap_top - hr < hy < cap_bottom + hr):
                
                # Check hand orientation (twisting motion)
                orientation = hand['orientation']
                if -45 < orientation < 45 or 135 < orientation < 225 or -135 < orientation < -45:
                    return 'open'  # Hand is in position to open/close
        
        return 'closed'
    
    def _handle_occlusions(self, bottles, hands):
        """Handle bottle occlusions by attaching to nearest hand"""
        for bottle in bottles:
            # Check if bottle is occluded (low confidence)
            if bottle['confidence'] < 0.5:
                # Find nearest hand
                nearest_hand = None
                min_dist = float('inf')
                
                for hand in hands:
                    dist = np.sqrt((bottle['position'][0] - hand['center'][0])**2 + 
                                  (bottle['position'][1] - hand['center'][1])**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_hand = hand
                
                # Attach bottle to nearest hand if close enough
                if nearest_hand and min_dist < 100:
                    # Update bottle position based on hand position
                    bottle['position'] = (
                        nearest_hand['center'][0] + bottle['size'][0] // 2,
                        nearest_hand['center'][1] + bottle['size'][1] // 2
                    )
                    bottle['attached_to_hand'] = nearest_hand['id']
                    bottle['occlusion_handled'] = True
        
        return bottles