# phygitalism test task

Task: https://github.com/phygitalism/test-task-point-cloud-analysis

## Install
```python
git clone https://github.com/denred0/phygitalism_test_task.git
cd phygitalism_test_task
pip install -r requirements.txt
```

## Split cloud points
Run script `split_cloud_points.py`
```python
python src/split_cloud_points.py %path_to_cloud% %split_type% 
python src/split_cloud_points.py data/fovea_tikal_guatemala_pcloud.asc voxel 
```

**Types:** 
<br>voxel
<br>dbscan

**Additional parameters:**
<br>_--radius_ - radius of cluster
<br>_--vis_ - show visualization
