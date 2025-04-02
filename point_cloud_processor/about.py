#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QTabWidget,
                           QTextEdit, QPushButton, QHBoxLayout, QWidget)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QPixmap, QFont, QDesktopServices


class AboutWidget(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Point Cloud Processor")
        self.setMinimumSize(600, 400)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # Create tabs
        tabs = QTabWidget()

        # Create About tab
        about_widget = QWidget()
        about_layout = QVBoxLayout(about_widget)

        # Application title
        title_label = QLabel("Point Cloud Processor")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        about_layout.addWidget(title_label)

        # Version
        version_label = QLabel("Version 1.0")
        version_label.setAlignment(Qt.AlignCenter)
        about_layout.addWidget(version_label)

        # Description
        desc_text = QTextEdit()
        desc_text.setReadOnly(True)
        desc_text.setHtml("""
        <p>Point Cloud Processor is an advanced application for processing and visualizing 3D point clouds.
        It provides tools for generating, manipulating, and visualizing point cloud data as well as
        performing surface reconstruction.</p>

        <p>Key features include:</p>
        <ul>
            <li>Point cloud generation (sphere, pyramid)</li>
            <li>Support for .pcd and .xyz file formats</li>
            <li>Normal estimation and orientation</li>
            <li>Poisson surface reconstruction</li>
            <li>Ball pivoting surface reconstruction</li>
            <li>Interactive 3D visualization</li>
            <li>Mesh export capabilities</li>
        </ul>

        <p>This application uses Open3D and PyVista for processing and visualization.</p>
        """)
        about_layout.addWidget(desc_text)

        tabs.addTab(about_widget, "About")

        # Create Creators tab
        creators_widget = QWidget()
        creators_layout = QVBoxLayout(creators_widget)

        # Credits title
        credits_label = QLabel("Development Team")
        credits_font = QFont()
        credits_font.setPointSize(14)
        credits_font.setBold(True)
        credits_label.setFont(credits_font)
        credits_label.setAlignment(Qt.AlignCenter)
        creators_layout.addWidget(credits_label)

        # Team members
        team_text = QTextEdit()
        team_text.setReadOnly(True)
        team_text.setHtml("""
        <h3>Team Members</h3>

        <p><b>John Doe</b> - Project Lead & Algorithm Developer</p>
        <p>Responsible for core algorithms implementation, point cloud processing pipeline,
        and overall architecture design.</p>

        <p><b>Jane Smith</b> - 3D Visualization Specialist</p>
        <p>Developed the interactive 3D visualization components and rendering optimizations.</p>

        <p><b>David Johnson</b> - UI/UX Designer</p>
        <p>Created the user interface design, application workflow, and user experience improvements.</p>

        <h3>Special Thanks</h3>
        <p>Special thanks to the Open3D and PyVista developer communities for their excellent libraries
        that made this application possible.</p>
        """)
        creators_layout.addWidget(team_text)

        tabs.addTab(creators_widget, "Creators")

        # Create Libraries tab
        libs_widget = QWidget()
        libs_layout = QVBoxLayout(libs_widget)

        # Libraries title
        libs_label = QLabel("Libraries & Dependencies")
        libs_label.setFont(credits_font)
        libs_label.setAlignment(Qt.AlignCenter)
        libs_layout.addWidget(libs_label)

        # Libraries list
        libs_text = QTextEdit()
        libs_text.setReadOnly(True)
        libs_text.setHtml("""
        <p>This application relies on the following open-source libraries:</p>

        <ul>
            <li><b>PyQt5</b> - Cross-platform GUI framework</li>
            <li><b>Open3D</b> - Library for 3D data processing</li>
            <li><b>NumPy</b> - Scientific computing library</li>
            <li><b>PyVista</b> - 3D visualization and mesh analysis toolkit</li>
            <li><b>pyvistaqt</b> - PyQt integration for PyVista</li>
        </ul>

        <p>Each of these libraries is subject to its own license terms.</p>
        """)
        libs_layout.addWidget(libs_text)

        tabs.addTab(libs_widget, "Libraries")

        main_layout.addWidget(tabs)

        # Buttons
        button_layout = QHBoxLayout()

        website_button = QPushButton("Visit Website")
        website_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://example.com")))

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)

        button_layout.addWidget(website_button)
        button_layout.addStretch()
        button_layout.addWidget(close_button)

        main_layout.addLayout(button_layout)
