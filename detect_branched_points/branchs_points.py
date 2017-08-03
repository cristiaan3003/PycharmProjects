import cv2
import ejemploWEB as rb
from skimage import img_as_ubyte

#================================================
#===============================================
#======================================================================
#======================================================================
if __name__ == "__main__":

    thin=cv2.imread("/home/asusn56/PycharmProjects/detect_branched_points/skeletos/img19.tiff")
    if len(thin.shape)==3:
        thin=cv2.cvtColor(thin, cv2.COLOR_BGR2GRAY)
    pruned_thin=rb.remove_branchs(thin,False)
    print(pruned_thin)
    cv2.imshow("thin",img_as_ubyte(pruned_thin))
    cv2.waitKey(0)
