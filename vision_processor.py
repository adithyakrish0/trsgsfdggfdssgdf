# vision_processor.py
import cv2
import numpy as np
from datetime import datetime
from hand_tracker import HandTracker
from bottle_tracker import BottleTracker

class VisionProcessor:
    def __init__(self):
        self.hand_tracker = HandTracker()
        self.bottle_tracker = BottleTracker()
        
    def process_frame(self, frame, session_id):
        """
        Process a single frame and return detected events.
        
        Args:
            frame: Input frame (BGR format).
            session_id: ID of the current vision session.
            
        Returns:
            List of detected events.
        """
        events = []
        
        # 1. Hand tracking
        hands = self.hand_tracker.track(frame)
        
        # 2. Bottle tracking (using your existing implementation)
        bottles = self.bottle_tracker.track(frame, None, hands)
        
        # 3. Event detection
        events = self.detect_events(hands, bottles, session_id)
        
        return events
    
    def detect_events(self, hands, bottles, session_id):
        """
        Detect high-level events like bottle opening, hand to mouth, etc.
        
        Args:
            hands: List of detected hands.
            bottles: List of detected bottles.
            session_id: ID of the current vision session.
            
        Returns:
            List of detected events.
        """
        events = []
        
        # Check for hand to mouth gesture
        for hand in hands:
            if self._is_hand_near_mouth(hand):
                events.append({
                    'type': 'hand_to_mouth',
                    'confidence': hand['confidence'],
                    'timestamp': datetime.now()
                })
        
        # Check for bottle opening/closing (using your existing implementation)
        for bottle in bottles:
            if bottle.get('state_changed', False):
                if bottle['state'] == 'open':
                    events.append({
                        'type': 'bottle_open',
                        'confidence': bottle['confidence'],
                        'timestamp': datetime.now()
                    })
                elif bottle['state'] == 'closed':
                    events.append({
                        'type': 'bottle_close',
                        'confidence': bottle['confidence'],
                        'timestamp': datetime.now()
                    })
        
        return events
    
    def _is_hand_near_mouth(self, hand):
        """
        Check if hand is near the mouth region.
        
        Args:
            hand: Hand information dictionary.
            
        Returns:
            Boolean indicating if hand is near mouth.
        """
        # Get hand center
        cx, cy = hand['center']
        
        # Estimate mouth position (above hand center by approximately hand radius)
        mouth_y = cy - hand['radius'] * 1.5
        
        # Check if hand is moving upward (toward mouth)
        vx, vy = hand['velocity']
        
        # Simple heuristic: hand is moving upward and is in the upper part of the frame
        if vy < -2 and cy < 480 * 0.6:  # Assuming frame height of 480
            return True
        
        return False
    
    def draw_debug_info(self, frame, hands, bottles):
        """
        Draw debug information on the frame.
        
        Args:
            frame: Input frame.
            hands: List of detected hands.
            bottles: List of detected bottles.
            
        Returns:
            Frame with debug information drawn.
        """
        # Draw hand information
        frame = self.hand_tracker.draw_debug_info(frame, hands)
        
        # Draw bottle information (using your existing implementation)
        for bottle in bottles:
            x, y = bottle['position']
            w, h = bottle['size']
            
            # Draw bottle rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw bottle state
            state = bottle.get('state', 'unknown')
            cv2.putText(frame, f"Bottle: {state}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame