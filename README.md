# Image Interpolation 

Image interpolation refers to the resizing of a digital image. Interpolation is the problem of approximating the value of a function for a non-given point in some space when given the value of that function in points around (neighboring) that point. 

Image interpolation have three main techniques:
### 1. Nearest Neighbor Interpolation
Nearest-neighbor interpolation (also known as proximal interpolation or, in some contexts, point sampling) is a simple method of multivariate interpolation in one or more dimensions. The nearest neighbor algorithm selects the value of the nearest point and does not consider the values of neighboring points at all, yielding a piecewise-constant interpolant. The algorithm is very simple to implement and is commonly used (usually along with mipmapping) in real-time 3D rendering to select color values for a textured surface ([Wikipedia](https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation#:~:text=Nearest%2Dneighbor%20interpolation%20(also%20known,in%20one%20or%20more%20dimensions.))).

!["Nearest Neighbor Interpolation"](https://user-images.githubusercontent.com/26917380/101096984-9cce3d00-35d5-11eb-821e-4e17dcb3b8ba.png)

### 2. Bilinear Interpolation
Bilinear interpolation is performed using linear interpolation first in one direction, and then again in the other direction. Bilinear interpolation uses values of only the 4 nearest pixels, located in diagonal directions from a given pixel, in order to find the appropriate color intensity values of that pixel ([Wikipedia](https://en.wikipedia.org/wiki/Bilinear_interpolation)).

!["Bilinear Interpolation"](https://user-images.githubusercontent.com/26917380/101096980-9c35a680-35d5-11eb-9a48-c7325c8a0e9c.png)
!["Bilinear Algorithm"](https://user-images.githubusercontent.com/26917380/101096977-9b047980-35d5-11eb-9756-8063cf5fc2be.jpeg)

### 3. Bicubic Interpolation 
In image processing, bicubic interpolation is often chosen over bilinear or nearest-neighbor interpolation in image resampling, when speed is not an issue. In contrast to bilinear interpolation, which only takes 4 pixels (2×2) into account, bicubic interpolation considers 16 pixels (4×4)([Wikipedia](https://en.wikipedia.org/wiki/Bicubic_interpolation#:~:text=In%20mathematics%2C%20bicubic%20interpolation%20is,interpolation%20or%20nearest%2Dneighbor%20interpolation.))
!["Bicubic Interpolation"](https://user-images.githubusercontent.com/26917380/101096971-99d34c80-35d5-11eb-861c-b8c88ab5866d.jpeg)
![ "Bicubic convolution formula"](https://user-images.githubusercontent.com/26917380/101096982-9c35a680-35d5-11eb-8ebc-9b2ccc0b4bf4.png)

## Results 
!["Opencv implementation"](https://user-images.githubusercontent.com/26917380/101097475-75c43b00-35d6-11eb-832d-a36a51241632.png)
!["My Implementation"](https://user-images.githubusercontent.com/26917380/101097462-7066f080-35d6-11eb-9fef-b62b58bc5fd1.png)
!["Compare results"](https://user-images.githubusercontent.com/26917380/101097490-7bba1c00-35d6-11eb-83a1-f4eb77df15bb.png)
## Clone project 
```
$ git clone https://github.com/yasinEnigma/Image_Interpolation/
$ pip3 install -r requirements.txt
$ python3 main.py

```

