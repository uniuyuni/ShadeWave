import math

# Scale factor between distances in uv space to a more user friendly "tint" parameter.
kTintScale = -3000.0

# Table from Wyszecki & Stiles, "Color Science", second edition, page 228.
kTempTable = [
    (0, 0.18006, 0.26352, -0.24341),
    (10, 0.18066, 0.26589, -0.25479),
    (20, 0.18133, 0.26846, -0.26876),
    (30, 0.18208, 0.27119, -0.28539),
    (40, 0.18293, 0.27407, -0.30470),
    (50, 0.18388, 0.27709, -0.32675),
    (60, 0.18494, 0.28021, -0.35156),
    (70, 0.18611, 0.28342, -0.37915),
    (80, 0.18740, 0.28668, -0.40955),
    (90, 0.18880, 0.28997, -0.44278),
    (100, 0.19032, 0.29326, -0.47888),
    (125, 0.19462, 0.30141, -0.58204),
    (150, 0.19962, 0.30921, -0.70471),
    (175, 0.20525, 0.31647, -0.84901),
    (200, 0.21142, 0.32312, -1.0182),
    (225, 0.21807, 0.32909, -1.2168),
    (250, 0.22511, 0.33439, -1.4512),
    (275, 0.23247, 0.33904, -1.7298),
    (300, 0.24010, 0.34308, -2.0637),
    (325, 0.24702, 0.34655, -2.4681),
    (350, 0.25591, 0.34951, -2.9641),
    (375, 0.26400, 0.35200, -3.5814),
    (400, 0.27218, 0.35407, -4.3633),
    (425, 0.28039, 0.35577, -5.3762),
    (450, 0.28863, 0.35714, -6.7262),
    (475, 0.29685, 0.35823, -8.5955),
    (500, 0.30505, 0.35907, -11.324),
    (525, 0.31320, 0.35968, -15.628),
    (550, 0.32129, 0.36011, -23.325),
    (575, 0.32931, 0.36038, -40.770),
    (600, 0.33724, 0.36051, -116.45)
]

class DngTemperature:
    def __init__(self):
        self.fTemperature = 0.0
        self.fTint = 0.0

    def set_xy_coord(self, xy):
        # Convert to uv space.
        u = 2.0 * xy[0] / (1.5 - xy[0] + 6.0 * xy[1])
        v = 3.0 * xy[1] / (1.5 - xy[0] + 6.0 * xy[1])
        
        last_dt = 0.0
        last_dv = 0.0
        last_du = 0.0
        
        for index in range(1, 31):
            # Convert slope to delta-u and delta-v, with length 1.
            du = 1.0
            dv = kTempTable[index][3]
            length = math.sqrt(1.0 + dv * dv)
            du /= length
            dv /= length
            
            # Find delta from black body point to test coordinate.
            uu = u - kTempTable[index][1]
            vv = v - kTempTable[index][2]
            
            # Find distance above or below line.
            dt = -uu * dv + vv * du
            
            # If below line, we have found line pair.
            if dt <= 0.0 or index == 30:
                if dt > 0.0:
                    dt = 0.0
                
                dt = -dt
                f = 0.0 if index == 1 else dt / (last_dt + dt)

                # Interpolate the temperature.
                self.fTemperature = 1.0E6 / (kTempTable[index - 1][0] * f + kTempTable[index][0] * (1.0 - f))
                
                # Find delta from black body point to test coordinate.
                uu = u - (kTempTable[index - 1][1] * f + kTempTable[index][1] * (1.0 - f))
                vv = v - (kTempTable[index - 1][2] * f + kTempTable[index][2] * (1.0 - f))
                
                # Interpolate vectors along slope.
                du = du * (1.0 - f) + last_du * f
                dv = dv * (1.0 - f) + last_dv * f
                
                length = math.sqrt(du * du + dv * dv)
                du /= length
                dv /= length
                
                # Find distance along slope.
                self.fTint = (uu * du + vv * dv) * kTintScale
                break
            
            # Try next line pair.
            last_dt = dt
            last_du = du
            last_dv = dv

    def get_xy_coord(self):
        result = [0.0, 0.0]
        
        # Find inverse temperature to use as index.
        r = 1.0E6 / self.fTemperature
        
        # Convert tint to offset in uv space.
        offset = self.fTint * (1.0 / kTintScale)
        
        # Search for line pair containing coordinate.
        for index in range(30):
            if r < kTempTable[index + 1][0] or index == 29:
                # Find relative weight of first line.
                f = (kTempTable[index + 1][0] - r) / (kTempTable[index + 1][0] - kTempTable[index][0])
                
                # Interpolate the black body coordinates.
                u = kTempTable[index][1] * f + kTempTable[index + 1][1] * (1.0 - f)
                v = kTempTable[index][2] * f + kTempTable[index + 1][2] * (1.0 - f)
                
                # Find vectors along slope for each line.
                uu1 = 1.0
                vv1 = kTempTable[index][3]
                uu2 = 1.0
                vv2 = kTempTable[index + 1][3]
                
                length1 = math.sqrt(1.0 + vv1 * vv1)
                length2 = math.sqrt(1.0 + vv2 * vv2)
                
                uu1 /= length1
                vv1 /= length1
                uu2 /= length2
                vv2 /= length2
                
                # Find vector from black body point.
                uu3 = uu1 * f + uu2 * (1.0 - f)
                vv3 = vv1 * f + vv2 * (1.0 - f)
                
                length3 = math.sqrt(uu3 * uu3 + vv3 * vv3)
                
                uu3 /= length3
                vv3 /= length3
                
                # Adjust coordinate along this vector.
                u += uu3 * offset
                v += vv3 * offset
                
                # Convert to xy coordinates.
                result[0] = 1.5 * u / (u - 4.0 * v + 2.0)
                result[1] = v / (u - 4.0 * v + 2.0)
                break
        
        return result
