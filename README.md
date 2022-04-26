# Physics-informed unsupervised deep learning for adaptive holographic imaging

We provied pytorch(python) and matlab implementations of **Physics-informed unsupervised deep learning for adaptive holographic imaging**. This code was written by **Chanseok Lee**.

# Overview
In optical imaging, image retrieval process often relies on inverse mapping between a measurement domain and an object domain. Deep learning methods have recently been shown to provide a fast and accurate framework to learn the inverse mapping. However, because the learning process solely depends on the statistical distribution of matched reference data, the reliability is in general compromised in practical imaging configurations where physical perturbations exist in various forms, such as mechanical movement and optical fluctuation. Here, in a holographic imaging scheme, we present a physics-informed unsupervised deep learning method that incorporates a parameterized forward physics model to render the adaptability in the inverse mapping even without any matched reference data for training. We show that both the morphology and the range of objects can be reconstructed under highly perturbative configurations where the object-to-sensor distance is set beyond the range of a given training data set. To prove the reliability of the proposed method in practical biomedical applications, we further demonstrate holographic imaging of red blood cells flowing in a cluster and diverse types of tissue sections presented without any ground truth data. Our results suggest that the physics-informed unsupervised approach effectively extends the adaptability of deep learning methods, and therefore, has great potential for solving a wide range of inverse problems in optical imaging techniques.  
<img src="/image/fig1.png" width="500" height="300">

# System Requirements
## Clone
```
git clone https://github.com/csleemooo/Physics_informed_unsupervised_deep_learning_for_adaptive_holographic_imaging
```

## Packages
The following libraries are necessary for running the codes.
- Python >= 3.7
- Pytorch >= 1.10.2
- phasepack == 1.5
- numpy
- PIL
- matplotlib

Please install requirements using below command.
```
pip install -r requirements.txt
```
which should install in about few minutes.

## Environements
The package development version is tested on windows. The developmental version of the package has been tested on the following systems and drivers.
- Windows 10
- CUDA 11.3
- cuDnn 8.2

# Demo
## Training
Train model with **MNIST** dataset.  
### Parameter description  
- num_depth: 1 or 5 (1 for single depth measurement, 5 for multiple depth measurements)  
- mode: phase, amplitude, complex_amplitude  
- result_root: The root where trained parameters and intermediate training results are saved.  
- Others: Other parameters (e.g. regularization constant, batch mode, batch size, iterations, and etc) can be modified. See ./model/Initialization.py. Also, distance range and the number of distance can be set by users. See train.py.
```
python train.py --data_name mnist --num_depth 5 --data_root DATA_PATH --result_root RESULT_PATH --mode complex_amplitude
```
Training models with 20000 iterations took up to 4 hours on a computer with 32 GB memory, Nvidia GeForce RTX 3080 Ti GPU, and 256GB solid-state drive.

### Example of training result
Intermediate training results are saved in './RESULT_ROOT/mnist_MODE_NUM_DEPTH/'. Example images are as follows. 

| ![iter100.png](/image/iter100.png)|![iter20000.png](/image/iter20000.png)|  
|:--:|:--:|
| *Iteration 100* | *Iteration 20000* |

  
## Testing
Test model with **MNIST** dataset. Complex amplitude of the sample reconstructed from single hologram intensity measurement can be compared with ground truth.  

### Test with MNIST
data_name: mnist
```
python test.py --data_name mnist --num_depth 5 --result_root RESULT_PATH --mode complex_amplitude
```

## Reproduce
Below commands reproduce the reported results from Fig 2 to Fig 5. 
Trained parameters used in this study can be downloaded from [here](https://drive.google.com/drive/folders/1Y6R8plKylzHNT4wkBEA4GeOreY9id1xm?usp=sharing.). Download .pth files and put them to **./model_parameters** folder 

### Test with experimental dataset
data_name: polystyrene_bead (num_depth: 1 or 6), tissue_array, or red_blood_cell
```
python test.py --data_name polystyrene_bead --num_depth 6 --result_root RESULT_PATH
python test.py --data_name tissue_array --result_root RESULT_PATH
```

