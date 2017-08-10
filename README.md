# Background Subtraction

A collection of background subtraction algorithms, implemented on top of Boost
and OpenCV. See the [COPYING](COPYING) for license rights and limitations (BSD).

## Algorithms

### Adaptive Median

It implements [1995McFarlane](#1995McFarlane). 

### Fuzzy Choquet

It implements [2008ElBaf](#2008ElBaf). It applies the Choquet integral for each pixel of
the image, to a triple of floating point values that represent:

- similarity between foreground and background LBP at that point, and
- similarity between two color planes of foreground and background, respectively.

where the color planes are notably not RGB (Choquet uses YCrCb).

### Fuzzy Sugeno

It implements [2006Zhang](#2006Zhang). Like above, but it uses a Sugeno integral in Ohta
colorspace.

### Sigma-Delta

It implements [2007Manzanera](#2007Manzanera).

### Temporal Median

It implements and approximation of [2006Calderara](#2006Calderara). The
divergence is in the use of a simpler pair of masks, with global thresholds.

## Utilities

There are a bunch of one-line internal utilities in the library that are useful
to me, mostly related to conversion between formats, scaling, etc. Find them in
`src/utils.cpp`.

I copied Eric Niebler's getline range code and adapted it to fetching frames
from a `cv::VideoCapture` object. It allows for some sweetness:

    cv::VideoCapture cap = ...
    for (const auto& frame : bs::getframes_from (cap)) {
        // ... use the frame
    }

The code for that is in `include/bs/frame_range.hpp`, enjoy.

## CMake and Windows

No.

## References

<a name="1995McFarlane">[1995McFarlane]</a> McFarlane, Nigel JB, and C. Paddy
Schofield. "Segmentation and tracking of piglets in images." *Machine vision and
applications* 8.3 (1995): 187-193.

<a name="2006Zhang">[2006Zhang]</a> Zhang, Hongxun, and De Xu. "Fusing color and
texture features for background model." *Fuzzy Systems and Knowledge Discovery*
(2006): 887-893.

<a name="2007Manzanera">[2007Manzanera]</a> Manzanera, Antoine, and Julien
C. Richefeu. "A new motion detection algorithm based on Σ–Δ background
estimation." *Pattern Recognition Letters* 28.3 (2007): 320-328.

<a name="2006Calderara">[2006Calderara]</a> Calderara, Simone, et al. "Reliable
background suppression for complex scenes." *Proceedings of the 4th ACM
international workshop on Video surveillance and sensor networks.* ACM, 2006.

<a name="2008ElBaf">[2008ElBaf]</a> El Baf, Fida, Thierry Bouwmans, and Bertrand
Vachon. "Fuzzy integral for moving object detection." Fuzzy
Systems, 2008. *FUZZ-IEEE 2008.(IEEE World Congress on Computational
Intelligence). IEEE International Conference on.* IEEE, 2008. 
