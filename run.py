#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Point Cloud Processor Launcher Script
"""

import sys
from point_cloud_processor.main import QApplication, MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
