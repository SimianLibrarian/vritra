# Vritra : a home-made reconstruction and segmentation algorithm

developed by Gaël Ginot during his PhD thesis @ Institut Charles Sadron, Strasbourg, Francce, 2018-2021
For inquiry, please contact me on my personal email : gael.ginot@protonmail.ch


This algorithm aims at the 3D reconstruction of foams and emulsions imaged using X-Ray tomography, starting from horizontal slices.

# List of steps for 3D reconstruction

# Importation and filtering
The horizontal slices are first imported and filtered to ease the segmentation in the next steps. The images are filtered with a bilateral filter (10.1145/1401132.1401134), which can be tuned with the spatial and gray value standard deviations spatial_kernel and intensity_kernel. The 2D version of the filter is applied once in every direction, before saving the filtered images.

# Binarisation 
The pictures are binarised according to the gray value of every pixel. For a gray value between imin and imax (tunable by user), the pixel value is set to 1, and 0 otherwise. The binarised slices are then stored before further usage.

# Segmentation
Thin films between bubbles and drops can be too thin for a simple segmentation. They are thus segmented using the watershed segmentation method first used by Lambert et al. (10.1016/j.colsurfa.2005.01.002). The centers of the bubbles/drops are detected bu using the maxima of an euclidean distance map , where every drop/bubble voxel is given a value corresponding to its distance to the closest voxel from the outer medium. In the case of very densely packed foams/emulsions, the number of peaks in a continuous body can be very high. To improve the precision of peak detection, we thus first perform a first segmentation, and perform the peak detection procedure on every labelled object individually. A second watershed segmentation is then performed to further segment touching bodies. The informations on the final bodies are then sotred in three different files : 1) volumes.csv storing, for every voxel (labelled by its (xyz) coordinates) its labels 2) contours.csv storing only the (xyz) position and label of the voxels of every body in contact with the outer medium 3) centroids.csv storing the position of the centroids of all the voxels belonging to every body.

# Contact detection
The contact detection is based on the surface-to-surface (s2s) distance detection described in Gaël Ginot PhD thesis (to be published). Test pairs are sorted by making a Delaunay triangulation of the space based on the positions of the centroids of the bubbles/drops, stored in centroids.csv. To detect the distance between the interfaces of the drops/bubbles, the distances between pairs of their contour voxels (stored in contours.csv) are computed. This computation is sped up 1) by selecting only the voxels inside a cylinder joining the two centroids 2) by using a C algorithm s2s.exe, stored in the s2s directory. From the distribution of the distances, the 5th centile is taken as the actual s2s distance to avoid wrongly labelled voxels. The radius of the cylinder and the choice of another centile can be changed by the user. Finally, a distance threshold is put in plae to decide if two drops/bubbles are in contact with respect to the distance between their surfaces.

# Packing characterisation
The contact network is used to determine geometrical and topological quantities in the packing : contact number Z, bond orientational parameters q4 and q6, pair correlation function g2r and Shannon entropy. All quantities are described in details in Gaël Ginot PhD thesis (o be published).


LICENSE
Text

Creative Commons License
The documentation and texts of the livestock resilience model for social-ecological modelling by Florian D. Schneider is licensed under a Creative Commons Attribution 4.0 International License.

Code

The MIT License (MIT)

Copyright (c) 2022 the authors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
About
