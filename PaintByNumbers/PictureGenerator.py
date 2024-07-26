import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import cv2

def CreateColorLabel(index, color):
    return str(index) + ", " + '#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))

def AdjustContrastBrightness(image, contrast, brightness):
    image = np.int16(image)
    image = image * (contrast/127+1) - contrast + brightness
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    return image

def OutlineColor(image, color):
    # convert to hsv colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hsv_color = cv2.cvtColor(np.reshape(color, (1,1,3)).astype('uint8'), cv2.COLOR_RGB2HSV)
    hsv_color = np.reshape(hsv_color, (3,))
    # find the colors within the boundaries
    mask = cv2.inRange(hsv, hsv_color, hsv_color)
    
    # Find contours from the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def GenerateContourLabels(contours, label):
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
    
def PerformKMeans(imagePath, resultPath, numClusters = 5, showColors = True, showResult = True, showNumbers = True, drawContours = True, addTransparency = 0.2):

    image = cv2.imread(imagePath)
    
    # Adjust constrast so colors are further away and can generate better cluster groups
    image = AdjustContrastBrightness(image, 50, 0)
    
    # cv2 imports images as BGR for some reason and HSV seems to generate the best kmeans results
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # reshape for k-means
    orig_shape = image.shape
    rows = image.shape[0]
    cols = image.shape[1]
    depth = image.shape[2]
    image = image.reshape((rows * cols, depth))
    
    # run k-means fit 
    kmeans = KMeans(n_clusters=numClusters, n_init="auto", random_state=5)
    kmeans.fit(image)

    # normalize compressed image
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)
    compressed_image = compressed_image.reshape(orig_shape)

    # labels are what compressed color goes in each pixel
    labels = list(kmeans.labels_)
    
    # centroid is alias for the k colors
    centroid = np.clip(kmeans.cluster_centers_.astype('uint8'), 0, 255)
    

    
    # Generate both contours and color number labels to place in center of contours
    # Numbers are placed in the centroid of it's contour
    if drawContours or showNumbers:
        contours = tuple()
        contour_labels = list()
        for color in range(len(centroid)):
            contour = OutlineColor(compressed_image, centroid[color])
            contours += contour
            contour_labels.append(GenerateContourLabels(contour, color + 1))
    
    #BGR is display format
    compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_HSV2BGR)
    
    # Transparency if you want it more of a paint guide
    if addTransparency < 1.0:
        white_image = np.full(orig_shape, 255, dtype='uint8')
        compressed_image = cv2.addWeighted(white_image, 0.1, compressed_image, addTransparency, 0)
    
    # Contours outline color regions if you want a more traditional paint by number
    if drawContours:
        compressed_image = cv2.drawContours(compressed_image, contours, -1, (0,0,0), 1)
    
    # show numbers separate because contours can be confusing to look at
    if showNumbers:
        for color in contour_labels:
            for contour_label in color:
                cv2.putText(compressed_image, str(contour_label[2]), (contour_label[0], contour_label[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # save compresed image
    _, image_name = os.path.split(imagePath)
    image_name, _ = image_name.split('.')
    cv2.imwrite(resultPath + image_name + '_' + str(numClusters) + '.png', compressed_image)

    # show final image
    if showResult:
        cv2.imshow(image_name, compressed_image)
        cv2.waitKey(0)

    # generate color guide wheel
    corrected_colors = cv2.cvtColor(kmeans.cluster_centers_.reshape((numClusters,1,3)).astype('uint8'), cv2.COLOR_HSV2RGB)
    corrected_colors = corrected_colors.reshape((numClusters,3))
    plt.rcParams['figure.figsize'] = (20, 12)
    percent = [1/len(labels)] * len(corrected_colors) #even color sizes
    pieLabels = [CreateColorLabel(x + 1, corrected_colors[x]) for x in range(len(corrected_colors))]
    plt.pie(percent, colors = np.array(corrected_colors/255), labels = pieLabels)
    plt.savefig(resultPath + image_name + '_' + str(numClusters) +'_color_guide.png')
    
    if showColors:
        plt.show()
        
