import math


class EuclideanDistTracker:
    def __init__(self, vel_med, err_rad):
        self.vel_med = vel_med
        self.err_rad = err_rad
        self.freq = {}
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def getstablepoints(self, freq_min):
        S = []
        for k in self.freq:
            if self.freq[k] >= freq_min:
                S.append(k)
        return S

    def getcenterpoints(self):
        return self.center_points.copy()

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                vx,vy = self.vel_med
                dist = math.hypot(cx - (pt[0]+vx), cy - (pt[1]+vy))
                if dist < self.err_rad:
                    self.center_points[id] = (cx, cy)
                    self.freq[id] += 1
                    #print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                self.freq[self.id_count] = 1
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        new_freq = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
            new_freq[object_id] = self.freq[object_id]

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        self.freq = new_freq.copy()
        return objects_bbs_ids



