import os
import sys
import numpy as np
import cv2
import statistics
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.uic import loadUi
from PyQt5.QtGui import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QLineEdit
#from google.colab.patches import cv2_imshow

class ShowImage(QMainWindow):
  def __init__(self):
    super(ShowImage, self).__init__()
    loadUi('GUITA.ui', self)
    self.image = None

    #Coba
    self.CobaButton.clicked.connect(self.Coba)
    self.Browse.clicked.connect(self.Cari)
    self.Load.clicked.connect(self.LoadImage)

  #PERCOBAAN
  def Cari(self):
    self.image, image_input = QFileDialog.getOpenFileName(self, 'Open File ', 'r' + 'C:\\',
                                          "Image Files(*.jpg *.jpeg)")

    imagePath = self.image[0]

    pixmap = QPixmap(imagePath)

    self.label.setPixmap(QPixmap(pixmap))
    self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    self.label.setScaledContents(True)

    print(imagePath)

  def LoadImage(self):
    image = cv2.imread(self.image)
    self.image = image
    self.displayImage(1)

#Metode Percobaan
  def Coba(self):
    #Input Gambar
    image_input = self.image

    #Mengubah dari BGR to RGB
    img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)


    #GaussianBlur
    gaussian_blur = cv2.GaussianBlur(img_rgb, (9, 9), 0)

    # SHARPENING 2D
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpning = cv2.filter2D(gaussian_blur, -1, filter)

    # K-MEANS
    vectorized = sharpning.reshape((-1, 3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    K = self.lineEdit.text()
    try:
      K = int(K)
      attempts = 10
      retval, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

      center = np.uint8(center)
      res = center[label.flatten()]

      kmean_result = res.reshape((sharpning.shape))

    except:
      QMessageBox.about(self, 'Error', 'Masukan Hanya Angka')
      pass

    #K = 6
    # attempts = 10
    # retval, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    #
    # center = np.uint8(center)
    # res = center[label.flatten()]
    #
    # kmean_result = res.reshape((sharpning.shape))


    figure_size = 20
    plt.figure(figsize=(figure_size, figure_size))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Gambar Asli')
    plt.xticks([]), plt.yticks([])
    #
    plt.subplot(1, 2, 2)
    plt.imshow(gaussian_blur)
    plt.title('Hasil Gaussian Blur')
    plt.xticks([]), plt.yticks([])
    plt.show()
    #
    figure_size = 15
    plt.figure(figsize=(figure_size, figure_size))
    plt.subplot(1, 2, 1)
    plt.imshow(sharpning)
    plt.title('Hasil Sharpening')
    plt.xticks([]), plt.yticks([])
    #
    plt.subplot(1, 2, 2)
    plt.imshow(kmean_result)
    plt.title('Segmentasi gambar setelah di klustering dengan k = %i' % K)
    plt.xticks([]), plt.yticks([])
    plt.show()

    # NILAI MEAN
    Means = cv2.mean(kmean_result)
    self.L_HasilMean.setText('Mean = ' + (str(Means))) #Menampilkan Hasil pada Label
    #print('Nilai Mean = ', Means_Int)


    # NILAI STDEV
    simpangan_baku = statistics.stdev(Means)
    self.L_HasilStdev.setText('Simpangan Baku = ' + (str(simpangan_baku))) #Menampilkan Hasil pada Label
    #print('Nilai Simpangan Baku = ', simpangan_baku)

    #Penetuan Kualitas Daging Berdasarkan Stdev
    if simpangan_baku >= 76 and simpangan_baku <= 85 :
      self.L_Daging.setText('Kualitas Daging :  Sangat Baik')
      #print('Daging Kualitas Baik')

    elif simpangan_baku >= 70 and simpangan_baku <= 75:
      self.L_Daging.setText('Kualitas Daging : Baik')

    elif simpangan_baku >=60 and simpangan_baku <= 69 :
      self.L_Daging.setText('Kualitas Daging : Buruk')

    elif simpangan_baku >=40 and simpangan_baku < 59 :
      self.L_Daging.setText('Kualitas Daging : Sangat Buruk')

    elif simpangan_baku < 40 and simpangan_baku > 85 :
      self.L_Daging.setText('Tidak Terdefinisi')

    #self.displayImage(1)



  def displayImage(self, windows=1):
    qformat = QImage.Format_Indexed8

    if len(self.image.shape) == 3:
      if (self.image.shape[2]) == 4:
        qformat = QImage.Format_RGBA8888
      else:
        qformat = QImage.Format_RGB888

    img = QImage(self.image,
                 self.image.shape[1],
                 self.image.shape[0],
                 self.image.strides[0],
                 qformat,
                 )
    img = img.rgbSwapped()


    #Frame
    if windows == 1:
      self.label.setPixmap(QPixmap.fromImage(img))
      self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
      self.label.setScaledContents(True)
    elif windows == 2:
      self.L_Gauss.setPixmap(QPixmap.fromImage(img))
      self.L_Gauss.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
      self.L_Gauss.setScaledContents(True)
    elif windows == 3:
      self.L_Sharp.setPixmap(QPixmap.fromImage(img))
      self.L_Sharp.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
      self.L_Sharp.setScaledContents(True)
    elif windows == 4:
      self.L_KMeans.setPixmap(QPixmap.fromImage(img))
      self.L_KMeans.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
      self.L_KMeans.setScaledContents(True)

app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Tugas AKhir')
window.show()
sys.exit(app.exec_())