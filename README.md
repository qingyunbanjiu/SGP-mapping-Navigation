# SGP-mapping-Navigation
Cross-Domain Aware Mapless Navigation Framework for Amphibious Robots Based on Sparse Gaussian Process Mapping
This project provides a Sparse Gaussian Process (SGP)-based dynamic mapping and D* Lite path planning framework for both simulation and real-world deployment, tailored for complex environments including land–water transition scenarios.
To run the simulation:
Install the required PyTorch version compatible with your system.
Clone this repository and ensure all dependencies (e.g., NumPy, matplotlib, etc.) are installed.
Download the sample environment point cloud data.
Run the following command:python d_star_lite_path_planning.py

Real-World Deployment. If you're using a real LiDAR device:
1. Modify the point cloud subscription topic in **sgp_mapping_node.py** to match your LiDAR's topic.
2. Before launching the LiDAR, start the SGP mapping process with: **roslaunch sgp_mapping_ros mapping.launch**
This will start building the terrain map based on your LiDAR input. You can visualize the map in RViz.
To enable navigation, use the following command to run the D* Lite planner: **rosrun dstar_navigation dstar_planner.py**
