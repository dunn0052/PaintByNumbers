import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io, exposure, filters, measure, morphology, segmentation
from skimage.color import rgb2gray, rgb2hsv
import os
import cv2
import colorsys

import plotly
import plotly.express as px
import plotly.graph_objects as go

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

class PictureGenerator:
    
    def __init__(self):
        pass
    
    def GenerateOutlines(self, image):
        #image = io.imread(imagePath)
        image = segmentation.clear_border(image)
        grayscale_image = rgb2gray(image)
        threshold = filters.threshold_otsu(grayscale_image)
        mask = grayscale_image > threshold
        #mask = morphology.remove_small_objects(mask, 50)
        #mask = morphology.remove_small_holes(mask, 50)
        labels = measure.label(mask)
        
        fig = px.imshow(grayscale_image, binary_string=True)
        fig.update_traces(hoverinfo='skip') # hover is only for label info

        '''
        props = measure.regionprops(labels, grayscale_image)
        properties = ['area', 'eccentricity', 'perimeter', 'intensity_mean']

        # For each label, add a filled scatter trace for its contour,
        # and display the properties of the label in the hover of this trace.
        for index in range(1, labels.max()):
            label_i = props[index].label
            contour = measure.find_contours(labels == label_i, 0.5)[0]
            y, x = contour.T
            hoverinfo = ''
            for prop_name in properties:
                hoverinfo += f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>'
            fig.add_trace(go.Scatter(
                x=x, y=y, name=label_i,
                mode='lines', fill='toself', showlegend=False,
                hovertemplate=hoverinfo, hoveron='points+fills'))
        '''
        plotly.io.show(fig)

        fig = px.imshow(image, binary_string=True)
    
    
    def SegmentImage(self, image, colors):
        
        image_hsv = rgb2hsv(image)
        colors = colors/255
        color = colorsys.rgb_to_hsv(colors[0], colors[0], colors[0])
        mask = cv2.inRange(image_hsv, color, color)
        result = cv2.bitwise_and(image, image, mask=mask)
        return result
    
    def CreateColorLabel(self, index, color):
        return str(index) + ", " + '#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))
    
    def ShowSingleColor(self, image, color):
        mask = cv2.inRange(image, color, color)
        masked = cv2.bitwise_and(image, image, mask=mask)
        plt.imshow(masked)
        plt.show()
        
    def DrawBorders(self, image):
        black_pixel = np.array([0,0,0])
        wroteLine = False
        for y in range(len(image) - 1):
            for x in range(len(image[y]) - 1):
                if not wroteLine and image[y][x][0] != image[y][x + 1][0]:
                    image[y][x] = black_pixel
                    wroteLine = True
                elif not wroteLine and image[y + 1][x][0] != image[y][x][0]:
                    image[y][x] = black_pixel
                else:
                    wroteLine = False
                    
                
        return image 
    
    def AdjustContrastBrightness(self, image, contrast, brightness):
        image = np.int16(image)
        image = image * (contrast/127+1) - contrast + brightness
        image = np.clip(image, 0, 255)
        image = np.uint8(image)
        return image
    
    def EnhanceContrast(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        l_channel, a, b = cv2.split(image)

        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))

        # Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

        return enhanced_img
    
    def GetHueArray(self, image):
        hueTable = list()
        hueRow = list()
        for idx in range(len(image)):
            for pixel in range(len(image[idx])):
                hueRow.append(image[idx][pixel][2])
            hueTable.append(hueRow)
            hueRow = list()
                
        return np.array(hueTable)
    
    def OutlineColor(self, image, color):
        # convert to hsv colorspace
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
        hsv_color = cv2.cvtColor(np.reshape(color, (1,1,3)).astype('uint8'), cv2.COLOR_RGB2HSV)
        hsv_color = np.reshape(hsv_color, (3,))
        # find the colors within the boundaries
        mask = cv2.inRange(hsv, hsv_color, hsv_color)
        
        # Find contours from the mask
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def GenerateContourLabels(self, contours, label):
        labels = list()
        for c in contours:
            # compute the center of the contour
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            
            if cv2.contourArea(c) < 100:
                continue
            
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            labels.append((cX, cY, label))
            
        return labels
    
    def SetAlpha(self, image, alpha_value):
        r_channel, g_channel, b_channel   = cv2.split(image)
        alpha_channel = np.ones(r_channel.shape, dtype=r_channel.dtype) * alpha_value
        image_RGBA = cv2.merge((r_channel, g_channel, b_channel, alpha_channel))
        return image_RGBA
        
        
    def PerformKMeans(self, imagePath, resultPath, numClusters = 5, showColors = True, showImage = True, showNumbers = True, drawContours = True, addTransparency = True):

        # pre process original image
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.AdjustContrastBrightness(image, 50, 0)
        #plt.imshow(image)  
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        #image = self.EnhanceContrast(image)
        # reshape for k-means
        orig_shape = image.shape
        rows = image.shape[0]
        cols = image.shape[1]
        depth = image.shape[2]
        image = image.reshape((rows * cols, depth))
        
        # run k-means fit 
        kmeans = KMeans(n_clusters=numClusters, n_init="auto", random_state=5)
        kmeans.fit(image)

        corrected_colors = cv2.cvtColor(kmeans.cluster_centers_.reshape((numClusters,1,3)).astype('uint8'), cv2.COLOR_HSV2RGB)
        corrected_colors = corrected_colors.reshape((numClusters,3))


        # normalize compressed image
        compressed_image = kmeans.cluster_centers_[kmeans.labels_]
        #compressed_image = corrected_colors[kmeans.labels_]
        compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)
        compressed_image = compressed_image.reshape(orig_shape)

        labels = list(kmeans.labels_)
        centroid = np.clip(kmeans.cluster_centers_.astype('uint8'), 0, 255)
        
        contours = tuple()
        contour_labels = list()
        
        for color in range(len(centroid)):
            contour = self.OutlineColor(compressed_image, centroid[color])
            contours += contour
            contour_labels.append(self.GenerateContourLabels(contour, color + 1))
        
        transparent_image = np.array(compressed_image, dtype=np.float)
        transparent_image /= 255.0
        a_channel = np.ones(orig_shape, dtype=np.float)/2.0
        transparent_image = transparent_image * a_channel
        
        compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_HSV2RGB)
        
        white_image = np.full(orig_shape, 255, dtype='uint8')
        
        
        if addTransparency:
            compressed_image = cv2.addWeighted(white_image, 0.7, compressed_image, 0.2, 0)
            
        if drawContours:
            compressed_image = cv2.drawContours(compressed_image, contours, -1, (0,0,0), 1)
        
        if showNumbers:
            for color in contour_labels:
                for contour_label in color:
                    cv2.putText(compressed_image, str(contour_label[2]), (contour_label[0], contour_label[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        
        compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)
        
             
        
        # save compresed image
        _, image_name = os.path.split(imagePath)
        image_name, _ = image_name.split('.')
        cv2.imwrite(resultPath + image_name + '_' + str(numClusters) + '.png', compressed_image)

        # show final image
        if showImage:
            cv2.imshow("Compressed Image", compressed_image)

        # Save color pallete 
        if showColors:
            plt.rcParams['figure.figsize'] = (20, 12)
            percent = [1/len(labels)] * len(corrected_colors) #even color sizes
            pieLabels = [self.CreateColorLabel(x + 1, corrected_colors[x]) for x in range(len(corrected_colors))]
            plt.pie(percent, colors = np.array(corrected_colors/255), labels = pieLabels)
            plt.savefig(resultPath + image_name + '_' + str(numClusters) +'_color_guide.png')
            plt.show()
            
