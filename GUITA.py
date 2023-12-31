# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUITA.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1131, 642)
        MainWindow.setStyleSheet("background-color: rgb(255, 240, 178);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 220, 281, 211))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.CobaButton = QtWidgets.QPushButton(self.centralwidget)
        self.CobaButton.setGeometry(QtCore.QRect(100, 530, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.CobaButton.setFont(font)
        self.CobaButton.setObjectName("CobaButton")
        self.L_HasilMean = QtWidgets.QLabel(self.centralwidget)
        self.L_HasilMean.setGeometry(QtCore.QRect(310, 370, 791, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.L_HasilMean.setFont(font)
        self.L_HasilMean.setInputMethodHints(QtCore.Qt.ImhHiddenText)
        self.L_HasilMean.setFrameShape(QtWidgets.QFrame.Box)
        self.L_HasilMean.setTextFormat(QtCore.Qt.AutoText)
        self.L_HasilMean.setAlignment(QtCore.Qt.AlignCenter)
        self.L_HasilMean.setObjectName("L_HasilMean")
        self.L_HasilStdev = QtWidgets.QLabel(self.centralwidget)
        self.L_HasilStdev.setGeometry(QtCore.QRect(470, 440, 461, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.L_HasilStdev.setFont(font)
        self.L_HasilStdev.setInputMethodHints(QtCore.Qt.ImhHiddenText)
        self.L_HasilStdev.setFrameShape(QtWidgets.QFrame.Box)
        self.L_HasilStdev.setAlignment(QtCore.Qt.AlignCenter)
        self.L_HasilStdev.setObjectName("L_HasilStdev")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(660, 310, 113, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(540, 310, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.L_Daging = QtWidgets.QLabel(self.centralwidget)
        self.L_Daging.setGeometry(QtCore.QRect(470, 520, 461, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.L_Daging.setFont(font)
        self.L_Daging.setInputMethodHints(QtCore.Qt.ImhHiddenText)
        self.L_Daging.setFrameShape(QtWidgets.QFrame.Box)
        self.L_Daging.setAlignment(QtCore.Qt.AlignCenter)
        self.L_Daging.setObjectName("L_Daging")
        self.Browse = QtWidgets.QPushButton(self.centralwidget)
        self.Browse.setGeometry(QtCore.QRect(30, 460, 91, 31))
        self.Browse.setObjectName("Browse")
        self.Load = QtWidgets.QPushButton(self.centralwidget)
        self.Load.setGeometry(QtCore.QRect(180, 460, 91, 31))
        self.Load.setObjectName("Load")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 180, 1141, 16))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.line.setFont(font)
        self.line.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line.setLineWidth(6)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setObjectName("line")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(400, 50, 411, 41))
        font = QtGui.QFont()
        font.setFamily("Nirmala UI")
        font.setPointSize(24)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(60, 0, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Nirmala UI")
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(10, 40, 241, 41))
        font = QtGui.QFont()
        font.setFamily("Nirmala UI")
        font.setPointSize(10)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(20, 80, 241, 41))
        font = QtGui.QFont()
        font.setFamily("Nirmala UI")
        font.setPointSize(10)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(10, 120, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Nirmala UI")
        font.setPointSize(10)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(410, 100, 411, 41))
        font = QtGui.QFont()
        font.setFamily("Nirmala UI")
        font.setPointSize(24)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1131, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Original Image"))
        self.CobaButton.setText(_translate("MainWindow", "Process"))
        self.L_HasilMean.setText(_translate("MainWindow", "Mean Result"))
        self.L_HasilStdev.setText(_translate("MainWindow", "Standard Deviation Result"))
        self.label_2.setText(_translate("MainWindow", "Insert K Value :"))
        self.L_Daging.setText(_translate("MainWindow", "Meat Quality"))
        self.Browse.setText(_translate("MainWindow", "Browse Image"))
        self.Load.setText(_translate("MainWindow", "Load Image"))
        self.label_3.setText(_translate("MainWindow", "Pendeteksi Kualitas Daging"))
        self.label_4.setText(_translate("MainWindow", "By Kelompok 9"))
        self.label_6.setText(_translate("MainWindow", "Muhammad Ariq Rifqy Yuzar (152020023)"))
        self.label_7.setText(_translate("MainWindow", "Nurochman Rizky Apriadi (152020114)"))
        self.label_8.setText(_translate("MainWindow", "Rayhan Maulana Herdiansyah (152020116)"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-style:italic;\">(Meat Quality Detector)</span></p></body></html>"))
import logo_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
