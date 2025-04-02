#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import tempfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QComboBox, QCheckBox, QSlider, QFileDialog,
                            QTabWidget, QSpinBox, QDoubleSpinBox, QLineEdit, QGroupBox,
                            QSplitter, QMessageBox, QStatusBar, QFrame)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QIcon, QFont

import open3d as o3d
import pyvista as pv
from pyvistaqt import QtInteractor

# Import other modules
from utils import (
    generate_sphere_point_cloud,
    generate_tetrahedron_point_cloud,
    load_point_cloud,
    orient_normals,
    poisson_reconstruction,
    ball_pivoting_reconstruction,
    o3d_to_pyvista
)
from about import AboutWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Cloud Processor")
        self.setMinimumSize(1200, 800)

        # Initialize settings
        self.settings = QSettings("PointCloudApp", "PointCloudProcessor")

        # Initialize variables
        self.point_cloud = None
        self.poisson_mesh = None
        self.ball_pivot_mesh = None

        # Set up the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Create the control panel
        self.setup_control_panel()

        # Create visualization panel
        self.setup_visualization_panel()

        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        # Set up the application style
        self.apply_stylesheet()

        # Show initial message
        self.show_info("Welcome to Point Cloud Processor! Select a point cloud source and configure processing options.")

    def setup_control_panel(self):
        # Create control panel widget
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_panel.setMaximumWidth(350)

        # Create input options group
        self.input_group = QGroupBox("Input Options")
        input_layout = QVBoxLayout()

        # Point cloud source selection
        source_label = QLabel("Point Cloud Source:")
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Generate Sphere", "Generate Pyramid", "Upload File"])
        self.source_combo.currentIndexChanged.connect(self.update_source_options)

        input_layout.addWidget(source_label)
        input_layout.addWidget(self.source_combo)

        # Number of points control (for generated point clouds)
        self.points_label = QLabel("Number of points:")
        self.points_slider = QSlider(Qt.Horizontal)
        self.points_slider.setMinimum(1000)
        self.points_slider.setMaximum(50000)
        self.points_slider.setValue(10000)
        self.points_slider.setTickInterval(5000)
        self.points_slider.setTickPosition(QSlider.TicksBelow)
        self.points_spin = QSpinBox()
        self.points_spin.setRange(1000, 50000)
        self.points_spin.setValue(10000)

        self.points_slider.valueChanged.connect(self.points_spin.setValue)
        self.points_spin.valueChanged.connect(self.points_slider.setValue)

        points_layout = QHBoxLayout()
        points_layout.addWidget(self.points_slider)
        points_layout.addWidget(self.points_spin)

        input_layout.addWidget(self.points_label)
        input_layout.addLayout(points_layout)

        # File upload control
        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        self.file_path_label = QLabel("No file selected")

        input_layout.addWidget(self.file_button)
        input_layout.addWidget(self.file_path_label)

        self.input_group.setLayout(input_layout)

        # Create processing options group
        self.processing_group = QGroupBox("Processing Options")
        processing_layout = QVBoxLayout()

        # Normal estimation controls
        self.normal_check = QCheckBox("Estimate and Orient Normals")
        self.normal_check.setChecked(True)
        self.normal_k_label = QLabel("Neighbors for normal estimation:")
        self.normal_k_spin = QSpinBox()
        self.normal_k_spin.setRange(5, 100)
        self.normal_k_spin.setValue(30)

        processing_layout.addWidget(self.normal_check)
        processing_layout.addWidget(self.normal_k_label)
        processing_layout.addWidget(self.normal_k_spin)

        self.normal_check.stateChanged.connect(self.toggle_normal_options)

        self.processing_group.setLayout(processing_layout)

        # Create reconstruction options group
        self.recon_group = QGroupBox("Reconstruction Options")
        recon_layout = QVBoxLayout()

        # Poisson reconstruction controls
        self.poisson_check = QCheckBox("Poisson Reconstruction")
        self.poisson_check.setChecked(True)
        self.poisson_depth_label = QLabel("Poisson depth:")
        self.poisson_depth_spin = QSpinBox()
        self.poisson_depth_spin.setRange(5, 12)
        self.poisson_depth_spin.setValue(8)

        recon_layout.addWidget(self.poisson_check)
        recon_layout.addWidget(self.poisson_depth_label)
        recon_layout.addWidget(self.poisson_depth_spin)

        self.poisson_check.stateChanged.connect(self.toggle_poisson_options)

        # Ball pivoting reconstruction controls
        self.ball_pivot_check = QCheckBox("Ball Pivoting Reconstruction")
        self.ball_pivot_check.setChecked(True)
        self.ball_radii_label = QLabel("Ball radii (comma separated):")
        self.ball_radii_edit = QLineEdit("1.0,2.0,4.0,8.0,16.0")

        recon_layout.addWidget(self.ball_pivot_check)
        recon_layout.addWidget(self.ball_radii_label)
        recon_layout.addWidget(self.ball_radii_edit)

        self.ball_pivot_check.stateChanged.connect(self.toggle_ball_pivot_options)

        self.recon_group.setLayout(recon_layout)

        # Process button
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_point_cloud)
        self.process_button.setMinimumHeight(40)

        # Save Button
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setMinimumHeight(40)
        self.save_button.setEnabled(False)

        # Add all widget groups to control layout
        self.control_layout.addWidget(self.input_group)
        self.control_layout.addWidget(self.processing_group)
        self.control_layout.addWidget(self.recon_group)
        self.control_layout.addWidget(self.process_button)
        self.control_layout.addWidget(self.save_button)

        # Add about button
        self.about_button = QPushButton("About / Credits")
        self.about_button.clicked.connect(self.show_about)
        self.control_layout.addWidget(self.about_button)

        # Add stretcher to push everything to the top
        self.control_layout.addStretch()

        # Add control panel to main layout
        self.main_layout.addWidget(self.control_panel)

        # Initial state update
        self.update_source_options()
        self.toggle_normal_options()
        self.toggle_poisson_options()
        self.toggle_ball_pivot_options()

    def setup_visualization_panel(self):
        # Create visualization panel widget
        self.visualization_panel = QWidget()
        vis_layout = QVBoxLayout(self.visualization_panel)

        # Create tabs for different visualizations
        self.visualization_tabs = QTabWidget()

        # Create tab for input point cloud
        self.input_tab = QWidget()
        input_layout = QVBoxLayout(self.input_tab)

        # Create plotter for input point cloud
        self.input_plotter_widget = QWidget()
        input_plotter_layout = QVBoxLayout(self.input_plotter_widget)
        self.input_plotter = QtInteractor(self.input_plotter_widget)
        input_plotter_layout.addWidget(self.input_plotter)
        input_layout.addWidget(self.input_plotter_widget)

        # Create info section for input
        self.input_info = QLabel("No point cloud loaded")
        self.input_info.setAlignment(Qt.AlignCenter)
        input_layout.addWidget(self.input_info)

        self.visualization_tabs.addTab(self.input_tab, "Input Point Cloud")

        # Create tab for normals
        self.normals_tab = QWidget()
        normals_layout = QVBoxLayout(self.normals_tab)

        # Create plotter for normals
        self.normals_plotter_widget = QWidget()
        normals_plotter_layout = QVBoxLayout(self.normals_plotter_widget)
        self.normals_plotter = QtInteractor(self.normals_plotter_widget)
        normals_plotter_layout.addWidget(self.normals_plotter)
        normals_layout.addWidget(self.normals_plotter_widget)

        # Create info section for normals
        self.normals_info = QLabel("No normals estimated yet")
        self.normals_info.setAlignment(Qt.AlignCenter)
        normals_layout.addWidget(self.normals_info)

        self.visualization_tabs.addTab(self.normals_tab, "Point Cloud with Normals")

        # Create tab for Poisson reconstruction
        self.poisson_tab = QWidget()
        poisson_layout = QVBoxLayout(self.poisson_tab)

        # Create plotter for Poisson
        self.poisson_plotter_widget = QWidget()
        poisson_plotter_layout = QVBoxLayout(self.poisson_plotter_widget)
        self.poisson_plotter = QtInteractor(self.poisson_plotter_widget)
        poisson_plotter_layout.addWidget(self.poisson_plotter)
        poisson_layout.addWidget(self.poisson_plotter_widget)

        # Create info section for Poisson
        self.poisson_info = QLabel("No Poisson reconstruction performed yet")
        self.poisson_info.setAlignment(Qt.AlignCenter)
        poisson_layout.addWidget(self.poisson_info)

        self.visualization_tabs.addTab(self.poisson_tab, "Poisson Reconstruction")

        # Create tab for Ball Pivoting reconstruction
        self.ball_pivot_tab = QWidget()
        ball_pivot_layout = QVBoxLayout(self.ball_pivot_tab)

        # Create plotter for Ball Pivoting
        self.ball_pivot_plotter_widget = QWidget()
        ball_pivot_plotter_layout = QVBoxLayout(self.ball_pivot_plotter_widget)
        self.ball_pivot_plotter = QtInteractor(self.ball_pivot_plotter_widget)
        ball_pivot_plotter_layout.addWidget(self.ball_pivot_plotter)
        ball_pivot_layout.addWidget(self.ball_pivot_plotter_widget)

        # Create info section for Ball Pivoting
        self.ball_pivot_info = QLabel("No Ball Pivoting reconstruction performed yet")
        self.ball_pivot_info.setAlignment(Qt.AlignCenter)
        ball_pivot_layout.addWidget(self.ball_pivot_info)

        self.visualization_tabs.addTab(self.ball_pivot_tab, "Ball Pivoting Reconstruction")

        # Add tabs to visualization layout
        vis_layout.addWidget(self.visualization_tabs)

        # Add visualization panel to main layout
        self.main_layout.addWidget(self.visualization_panel)

    def update_source_options(self):
        source = self.source_combo.currentText()

        # Show/hide appropriate controls based on source
        if source in ["Generate Sphere", "Generate Pyramid"]:
            self.points_label.setVisible(True)
            self.points_slider.setVisible(True)
            self.points_spin.setVisible(True)
            self.file_button.setVisible(False)
            self.file_path_label.setVisible(False)
        else:  # Upload File
            self.points_label.setVisible(False)
            self.points_slider.setVisible(False)
            self.points_spin.setVisible(False)
            self.file_button.setVisible(True)
            self.file_path_label.setVisible(True)

    def toggle_normal_options(self):
        enabled = self.normal_check.isChecked()
        self.normal_k_label.setEnabled(enabled)
        self.normal_k_spin.setEnabled(enabled)

    def toggle_poisson_options(self):
        enabled = self.poisson_check.isChecked()
        self.poisson_depth_label.setEnabled(enabled)
        self.poisson_depth_spin.setEnabled(enabled)

    def toggle_ball_pivot_options(self):
        enabled = self.ball_pivot_check.isChecked()
        self.ball_radii_label.setEnabled(enabled)
        self.ball_radii_edit.setEnabled(enabled)

    def select_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Point Cloud File", "",
            "Point Cloud Files (*.pcd *.xyz);;All Files (*)",
            options=options
        )
        if file_path:
            self.file_path_label.setText(os.path.basename(file_path))
            self.file_path_label.setToolTip(file_path)

    def process_point_cloud(self):
        try:
            self.statusBar.showMessage("Processing...")

            # Get source type
            source = self.source_combo.currentText()

            # Generate or load point cloud
            if source == "Generate Sphere":
                num_points = self.points_spin.value()
                self.point_cloud = generate_sphere_point_cloud(num_points=num_points)
                self.show_info(f"Generated sphere with {len(self.point_cloud.points)} points")
            elif source == "Generate Pyramid":
                num_points = self.points_spin.value()
                self.point_cloud = generate_tetrahedron_point_cloud(num_points=num_points)
                self.show_info(f"Generated pyramid with {len(self.point_cloud.points)} points")
            else:  # Upload File
                file_path = self.file_path_label.toolTip()
                if not file_path:
                    self.show_error("Please select a file first")
                    return

                try:
                    self.point_cloud = load_point_cloud(file_path)
                    self.show_info(f"Loaded point cloud with {len(self.point_cloud.points)} points")
                except Exception as e:
                    self.show_error(f"Error loading file: {str(e)}")
                    return

            # Visualize input point cloud
            self.visualize_input()

            # Process normals if requested
            if self.normal_check.isChecked():
                normal_k = self.normal_k_spin.value()
                self.point_cloud = orient_normals(self.point_cloud, k=normal_k)
                self.visualize_normals(normal_k)

            # Perform Poisson reconstruction if requested
            if self.poisson_check.isChecked():
                poisson_depth = self.poisson_depth_spin.value()
                self.poisson_mesh = poisson_reconstruction(self.point_cloud, depth=poisson_depth)
                self.visualize_poisson(poisson_depth)

            # Perform Ball Pivoting reconstruction if requested
            if self.ball_pivot_check.isChecked():
                try:
                    ball_radii = [float(r.strip()) for r in self.ball_radii_edit.text().split(",")]
                    self.ball_pivot_mesh = ball_pivoting_reconstruction(self.point_cloud, radii=ball_radii)
                    self.visualize_ball_pivot(ball_radii)
                except Exception as e:
                    self.show_error(f"Ball Pivoting failed: {str(e)}")

            # Enable save button if any processing was successful
            self.save_button.setEnabled(True)

            self.statusBar.showMessage("Processing complete", 3000)

        except Exception as e:
            self.show_error(f"Processing error: {str(e)}")

    def visualize_input(self):
        # Clear previous plot
        self.input_plotter.clear()

        # Convert to PyVista PolyData and add to plotter
        pv_mesh = o3d_to_pyvista(self.point_cloud)
        self.input_plotter.add_mesh(pv_mesh, point_size=5, render_points_as_spheres=True, color='lightblue')

        # Set view and background
        self.input_plotter.view_isometric()
        self.input_plotter.background_color = 'white'
        self.input_plotter.reset_camera()

        # Update info
        info_text = f"Number of points: {len(self.point_cloud.points)}\n"
        info_text += f"Has normals: {'Yes' if self.point_cloud.has_normals() else 'No'}\n"
        info_text += f"Has colors: {'Yes' if self.point_cloud.has_colors() else 'No'}"
        self.input_info.setText(info_text)

        # Switch to input tab
        self.visualization_tabs.setCurrentIndex(0)

    def visualize_normals(self, normal_k):
        # Clear previous plot
        self.normals_plotter.clear()

        # Convert to PyVista PolyData and add to plotter
        pv_mesh = o3d_to_pyvista(self.point_cloud)
        self.normals_plotter.add_mesh(pv_mesh, point_size=5, render_points_as_spheres=True, color='lightblue')

        # Add normals as arrows
        if self.point_cloud.has_normals():
            self.normals_plotter.add_mesh(
                pv_mesh.glyph(orient="Normals", scale=False, factor=0.05),
                color='red'
            )

        # Set view and background
        self.normals_plotter.view_isometric()
        self.normals_plotter.background_color = 'white'
        self.normals_plotter.reset_camera()

        # Update info
        info_text = f"Normals estimation complete\n"
        info_text += f"Used {normal_k} nearest neighbors"
        self.normals_info.setText(info_text)

    def visualize_poisson(self, poisson_depth):
        # Clear previous plot
        self.poisson_plotter.clear()

        if self.poisson_mesh and self.poisson_mesh.has_triangles():
            # Convert to PyVista PolyData and add to plotter
            pv_mesh = o3d_to_pyvista(self.poisson_mesh)
            self.poisson_plotter.add_mesh(pv_mesh, color='lightblue', smooth_shading=True, show_edges=True)

            # Set view and background
            self.poisson_plotter.view_isometric()
            self.poisson_plotter.background_color = 'white'
            self.poisson_plotter.reset_camera()

            # Update info
            info_text = f"Number of triangles: {len(self.poisson_mesh.triangles)}\n"
            info_text += f"Reconstruction depth: {poisson_depth}"
            self.poisson_info.setText(info_text)
        else:
            self.poisson_info.setText("Poisson reconstruction failed: no triangles generated")

    def visualize_ball_pivot(self, ball_radii):
        # Clear previous plot
        self.ball_pivot_plotter.clear()

        if self.ball_pivot_mesh and self.ball_pivot_mesh.has_triangles():
            # Convert to PyVista PolyData and add to plotter
            pv_mesh = o3d_to_pyvista(self.ball_pivot_mesh)
            self.ball_pivot_plotter.add_mesh(pv_mesh, color='lightblue', smooth_shading=True, show_edges=True)

            # Set view and background
            self.ball_pivot_plotter.view_isometric()
            self.ball_pivot_plotter.background_color = 'white'
            self.ball_pivot_plotter.reset_camera()

            # Update info
            info_text = f"Number of triangles: {len(self.ball_pivot_mesh.triangles)}\n"
            info_text += f"Used radii: {ball_radii}"
            self.ball_pivot_info.setText(info_text)
        else:
            self.ball_pivot_info.setText("Ball Pivoting reconstruction failed: no triangles generated")

    def save_results(self):
        if not self.point_cloud:
            self.show_error("No point cloud to save")
            return

        options = QFileDialog.Options()
        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Results", "", options=options)

        if save_dir:
            try:
                # Save original point cloud
                o3d.io.write_point_cloud(os.path.join(save_dir, "point_cloud.pcd"), self.point_cloud)

                # Save meshes if available
                if self.poisson_mesh and self.poisson_mesh.has_triangles():
                    o3d.io.write_triangle_mesh(os.path.join(save_dir, "poisson_mesh.obj"), self.poisson_mesh)

                if self.ball_pivot_mesh and self.ball_pivot_mesh.has_triangles():
                    o3d.io.write_triangle_mesh(os.path.join(save_dir, "ball_pivot_mesh.obj"), self.ball_pivot_mesh)

                self.show_info(f"Results saved to {save_dir}")

            except Exception as e:
                self.show_error(f"Error saving results: {str(e)}")

    def show_about(self):
        about_dialog = AboutWidget(self)
        about_dialog.exec_()

    def show_info(self, message):
        self.statusBar.showMessage(message, 3000)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.statusBar.showMessage("Error occurred", 3000)

    def apply_stylesheet(self):
        # Set application font
        app_font = QFont("Segoe UI", 9)
        QApplication.setFont(app_font)

        # Apply some basic styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:pressed {
                background-color: #0a6fc2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 12px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom: none;
            }
            QTabBar::tab:!selected {
                background-color: #dddddd;
            }
            QLabel {
                color: #333333;
            }
            QCheckBox {
                spacing: 5px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 4px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
