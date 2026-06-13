from roboflow import Roboflow
rf = Roboflow(api_key="maCXBS53AXaqghjUejsq")
project = rf.workspace("bangga").project("ppe-0.3")
version = project.version(1)
dataset = version.download("yolov8")
print("Dataset downloaded successfully!")
print("Check the folder - it has images and labels")