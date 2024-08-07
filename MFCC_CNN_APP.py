
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow


from UI import Ui_MainWindow

orders_dic = {
    'Takeoff': 0,
    'Landing': 1,
    'Advance': 2,
    'Retreat': 3,
    'Rise': 4
}

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    # 显示主窗口
    MainWindow.show()


    # 运行应用程序
    sys.exit(app.exec_())





