# Point Cloud Processor

A PyQt-based application for processing and visualizing 3D point clouds with surface reconstruction capabilities.

## Features

- Generate point clouds of basic shapes (sphere, pyramid)
- Load point clouds from PCD and XYZ files
- Normal estimation and orientation
- Surface reconstruction using:
  - Poisson Reconstruction
  - Ball Pivoting Reconstruction
- Interactive 3D visualization with PyVista
- Exports meshes to OBJ format

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/point-cloud-processor.git
   cd point-cloud-processor
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```
python run.py
```

### Processing a Point Cloud

1. Select a point cloud source (generate or load from file)
2. Configure processing options
3. Click "Process" button
4. View the results in the visualization tabs
5. Save results if desired

## Dependencies

- PyQt5: GUI framework
- Open3D: Point cloud processing library
- NumPy: Numerical computing
- PyVista: 3D visualization
- pyvistaqt: PyVista integration with Qt

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- PRIYANSHU TIWARI - Project Lead & Algorithm Developer
- N RAM SRUJAN RAJ - 3D Visualization Specialist
- RISHABH RAJ - UI/UX Designer
- ADITYA KUMAR - System App Devloper

## Mentor
- Dr S AMBAREESH (Profesor of AIML Dept at SIR M VISVEVARAYA INSTITUTE OF TECHNOLOGY)