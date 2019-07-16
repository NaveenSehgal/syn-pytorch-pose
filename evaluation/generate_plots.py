''' 
Generate plots for paper.

Inputs:
- dataset name: {mpii, ScanAva, AC2D, SJL}
- models: list of model paths and names

Outputs:
- Plot of accuracy vs PCKh@x for different models over designated dataset

================= Steps ===================
1. Generate detections.mat and detections_our_format.mat for ScanAva
2. Write pipeline that:
    a. Evaluate model X on dataset A
    b. Reads results of a) and calculates PCKh array
    c. Continues for all models and generates final plot for dataset A
'''



