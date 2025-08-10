#  Smooth Vector-Field Drone Simulation

This project is a **2D drone navigation simulator** using **OpenCV** and a **vector-field based local planner** with:
- Smooth, realistic turning dynamics (yaw rate & acceleration limits)
- Continuous vector blending for goal attraction & obstacle repulsion
- Wall-following with hysteresis (no jitter near obstacles)
- Collision-safe movement using triangle body geometry
- Configuration-space inflated obstacles to prevent clipping
- Sub-step motion integration to avoid tunneling through corners
- Goal detection with dwell time for a clean stop


---

##  Features
- **Vector Field Navigation** – continuous steering toward a goal while repelling from obstacles
- **Hysteresis Wall-Following** – keeps a consistent side when skirting obstacles
- **Inflated Obstacles** – ensures clearance for the drone’s footprint
- **Sub-Step Collision Checks** – prevents passing through edges at high speeds
- **Adjustable Dynamics** – max yaw rate, yaw acceleration, and smoothness tuning
- **Goal Dwell Stop** – prevents overshooting or circling at the goal

---

##  Requirements
- Python 3.8+
- [OpenCV](https://pypi.org/project/opencv-python/)
- NumPy

Install dependencies:
```bash
pip install opencv-python numpy
```

---

##  Usage
Clone this repository and run:
```bash
python drone_predictive_smart_planner.py
```

Controls:
- Press `ESC` to quit the simulation.

The drone will automatically:
1. Start from the bottom center
2. Navigate toward the goal (blue circle)
3. Avoid static obstacles (red blocks)


---

## 📄 License
MIT License – feel free to use and modify.

---

## 🛠️ Author
Developed by **Parham Fatemi**  
