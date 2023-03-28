# 3D-ViSa
This is a 3D visual saliency dataset containing 540 view-dependent saliency maps for 180 3D meshes. All data can be downloaded via the following links.

[3D models](https://drive.google.com/file/d/1EDrzJaQQs6_oDR2GyMXUFMceOdSj3gn9/view?usp=sharing) - 180 3D meshes used to acquire the fixation maps.

[2D views of 3D models](https://drive.google.com/file/d/1ZrvMnEIJ-yVzKKQZe7nksV4nraDP8JT8/view?usp=share_link) - 540 rendered 2D views of the 3D meshes. Each mesh has 3 views.

[Viewpoints for generating 2D views](https://drive.google.com/file/d/1FI19eqOlXFyELG9N9nC1vKQMKzRQU2sP/view?usp=sharing) - 180 files recording the viewpoints from which the 2D views of the 3D meshes are generated. Each file contains a 3x2 matrix and each row corresponds to a viewpiont represented in the form of (azimuth, elevation) in degrees.

[2D saliency maps for 2D views](https://drive.google.com/file/d/1IYPmHzV7RMVVuDL2OcuA5ag7cCfp8a1H/view?usp=sharing) - 540 2D saliency maps for the 2D views of the 3D meshes. Each saliency map in the size of 1200x1200 is generated by blurring the fixation map containing all raw fixations with a Gaussian filter of size 100 and standard deviation 30.

[3D saliency maps for 3D meshes](https://drive.google.com/file/d/1SphvQhkxwM7ok6vNVeH--ii7z_gP6XdI/view?usp=sharing) - 540 view-dependent 3D saliency maps for the 3D meshes. Each 3D saliency map recording the per-vertex saliency values is generated by mapping the corresponding 2D saliency map onto the surface of the 3D mesh while considering the visibility of each 3D vertex with regard to the corresponding viewpoint. Here we provide the MATLAB codes for doing so. Once you downlonad and unzip the above data and files, simply run 'create_3dsaliency.m' to generate the 3D saliency maps.   
