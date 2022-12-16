# Digital Image Processing 
Assignment #4

Due: Tue 12/02/22 11:59 PM

**Filtering:**

Write code for computing Median, Arithmetic Mean, Geometric Mean, Adaptive Local Noise reduction and Adaptive Median filters. 
The input to your program is a 2D matrix.

  - Starter code available in directory Denoise/
      - \__init__(): Will intialize the required variable for filtering (image, filter_name, filter_size). There is no need to edit this function  
  - Denoise/Filtering.py: Edit the functions 'get_median_filter', 'get_arithmetic_mean', 'get_geometric_mean', 'get_local_noise', and 'get_adaptive_mean'. you are welcome to add more function.
  - For this part of the assignment, please implement your own code for all computations, do not use built-in functions, like "medianBlur", "MaxFilter", "numpy.pad" from PIL, opencv or other libraries - that directly accomplish the objective of the question. You can use any math (import math) related functions such as "prod", "pow" and "sum".
    You can also make use python builtin functions such as "sorted" for order statistic filters.   
  
filtering(): Write your code to perform image denoising/filtering here using the previous implemented filters. The steps can be used as a guideline for filtering. All the variable have already been initialized and can be used as self.image, self.filter_name, etc. The variable self.filter is a handle to each of the five filters functions. 
  - The function return the denoised image.
  - This part of the assignment can be run using dip_hw_filter.py (there is no need to edit this file)
  - Usage: 
  
        ./dip_hw_filter.py -i Lenna.png -f arithmetic_mean -n gaussian
        python dip_hw_filter.py -i Lenna.png -f arithmetic_mean -n gaussian
        
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (dip_hw_filter.py automatically does this)
  - Two images are provided for testing: Lenna.png and Lenna0.jpg
  
---
**Grayscale to color conversion:**

Write code to convert a grayscale image into a color image using the two techniques covered in class: Color slicing and Intensity to color tranformation using rectified sine wave functions. 
The input to your program is a 2D matrix.

  - Starter code available in directory Coloring/
  - Coloring/Coloring.py: Edit the functions 'intensity_slicing', and 'color_transformation' you are welcome to add more function.
  - For this part of the assignment, please implement your own code for all computations, do not use built-in functions  from PIL, opencv or other libraries - that directly accomplish the objective of the question. You can use math and random related functions.
 
    
color_slicing(image, n_slices):
    - Write code to tranform greyscale image to color image using intensity slicing. 
    - The function returns the colored image.

color_transformation(image, n_slices, theta): 
    - Write code to tranform greyscale image to color image using intensity to color transformation using rectified sin waves.
    - The function returns the colored image.

  - This part of the assignment can be run using dip_hw_color.py (there is no need to edit this file)
  - Usage: 
  
        ./dip_hw_color -i cat.jpg -n 5
        python dip_hw_color.py -i cat.jpg -n 5
        
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (dip_hw_color.py automatically does this)
  - Multiple images are available for testing (cat.jpg, Medical.PNG, pluto.jpg, and luggage.jpeg)
  
--

  
PS. Files not to be changed: requirements.txt and Jenkins file 
  
1. Any output file or image should be written to output/ folder

The TA will only be able to see your results if these condition is met

1. Filtering       - 60 Pts.
2. Coloring        - 40 Pts.
 
    Total          - 100 Pts.

---
<sub><sup>License: Property of Quantitative Imaging Laboratory (QIL), Department of Computer Science, University of Houston.
This software is property of the QIL, and should not be distributed, reproduced, or shared online, without the permission of the author
This software is intended to be used by students of the digital image processing course offered at University of Houston.
The contents are not to be reproduced and shared with anyone with out the permission of the author.
The contents are not to be posted on any online public hosting websites without the permission of the author.
The software is cloned and is available to the students for the duration of the course.
At the end of the semester, the Github organization is reset and hence all the existing repositories are reset/deleted, to accommodate the next batch of students.</sub></sup>
