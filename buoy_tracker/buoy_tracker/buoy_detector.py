from .EM import *
import imutils
import cv2
import numpy as np

class BuoyDetector:
    def __init__(self, weights_path, test_path) -> None:
        self.buoy_colors = ['orange','green','yellow']
        self.training_channels={'orange':(1,2),'green':(0,1),'yellow':(1,2)}

        self.K = 3
        self.Theta = {}
        
        for color in self.buoy_colors:
            Sigma, mu, pi = readGMM(weights_path, color)
            self.Theta[color] = {'Sigma':Sigma,'mu':mu,'pi':pi}

        train_percent = .6
        self.K = 3
        self.threshold = determineThesholds(test_path, self.Theta, self.buoy_colors, self.K, train_percent, self.training_channels)
        print(self.threshold)
    
    def findContours(self, masked_image, segmented_frames):
        grey = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(grey, (5,5), 0)
        contours= cv2.findContours(blurred_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        if len(contours) == 0:
            return None
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:8]

        circle_inds=[]
        contour_valid_thresh = 0.4
        contour_info = []
        for i, contour in enumerate(sorted_contours):
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            contour_info.append((x, y, radius))
            A=math.pi*radius**2
            A_contour=cv2.contourArea(contour)
            if abs(A-A_contour) / A < contour_valid_thresh:
                circle_inds.append(i)
        detected_buoys = {}
        for ind in circle_inds:
            # buoy_contour=sorted_contours[ind]
            (x, y, radius) = contour_info[ind]
            A=math.pi*radius**2
            count=[]
            if A>500:
                for color in self.buoy_colors:
                    buoy_region = segmented_frames[color][
                                int(y-radius):int(y+radius),
                                int(x-radius):int(x+radius),:]
                    grey = cv2.cvtColor(buoy_region, cv2.COLOR_BGR2GRAY)
                    count.append(len(np.nonzero(grey)[0]))

                max_ind = count.index(max(count))
                if self.buoy_colors[max_ind] in detected_buoys:
                    detected_buoys[self.buoy_colors[max_ind]].append((x, y, radius))
                else:
                    detected_buoys[self.buoy_colors[max_ind]] = [(x, y, radius)]
        return detected_buoys


    def detect(self, frame):
        h,w = frame.shape[:2]
        segmented_frames = {}

        for color in self.buoy_colors:
            segmented_frames[color] = np.zeros([h,w,3],np.uint8)

        probs = {}
        for color in self.buoy_colors:
            Sigma = self.Theta[color]['Sigma']
            mu = self.Theta[color]['mu']
            pi = self.Theta[color]['pi']

            ch1 = frame[:, :, self.training_channels[color][0]].flatten()
            ch2 = frame[:, :, self.training_channels[color][1]].flatten()

            x = []
            # TODO: replace this with better numpy code
            for ch_x,ch_y in zip(ch1,ch2):
                x_i = np.array([ch_x,ch_y])
                x.append(x_i)
        
            x = np.asarray(x)

            K = len(mu)
            p = np.zeros((1,len(x)))
            for k in range(K):
                p += multivariate_normal.pdf(x, mean=mu[k], cov = Sigma[k])*pi[k]

            probs[color] = p.T

        for i in range(len(x)):
            pixel_p = []
            
            for color in self.buoy_colors:
                pixel_p.append(probs[color][i])
            
            max_ind = pixel_p.index(max(pixel_p))

            row = i//w
            column = i%w

            if max_ind == 0 and pixel_p[max_ind] > self.threshold['orange']:
                segmented_frames['orange'][row, column] = (14,127,255)
            elif max_ind == 1 and pixel_p[max_ind] > self.threshold['green']:
                segmented_frames['green'][row, column] = (96,215,30)
            elif max_ind == 2 and pixel_p[max_ind] > self.threshold['yellow']:
                segmented_frames['yellow'][row, column] = (77,245,255)
        
        all_buoy_masked = np.zeros([h, w, 3], np.uint8)
        for color in self.buoy_colors:
            all_buoy_masked = cv2.bitwise_or(all_buoy_masked, segmented_frames[color])

        detected_buoys = self.findContours(all_buoy_masked, segmented_frames)
        return detected_buoys

    def draw_buoy(self, frame, detected_buoys):
        bgr_colors={'orange':(14,127,255),'green':(96,215,30),'yellow':(77,245,255)}

        if detected_buoys is None:
            return frame
        for color in detected_buoys:
            ring_color = bgr_colors[color]
            for (x, y, radius) in detected_buoys[color]:
                cv2.circle(frame, (int(x), int(y)), int(radius), ring_color, 2)
        return frame