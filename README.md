# CVCS_

The primary goal of this project was to enhance a custom-designed autonomous lawn-robot system
initially developed by one of our team members. At its inception, the robot could navigate open environments using LiDAR to detect obstacles
and randomly adjust its direction, but it didn’t exploit the CMOS sensor fixed on top of the system.
Recognizing its potential for improvement, we developed a more advanced software system utilizing computer vision techniques to better guide the robot and
introduce new features. A critical improvement is in the new capability of detecting both the state of the grass in front of the robot and the presence of
a nearby occlusion, allowing better conduced navigation. Additionally, we introduced the capability to distinguish familiar environments from new ones, enabling the robot to adapt its navigation strategy by constructing a new representation of the surroundings, daily optimizing times and energy.

The .pdf file "Final Report" shows the project more in detail

## Citazione

A part of the code for this project was taken from [Occupancy Anticipation for Efficient Exploration and Navigation](https://github.com/facebookresearch/OccupancyAnticipation.git), which refers to the work of:

Ramakrishnan, S. K., Al-Halah, Z., & Grauman, K. (2020). "Occupancy Anticipation for Efficient Exploration and Navigation". In *Proceedings of the European Conference on Computer Vision (ECCV)*.
