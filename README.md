# Physics-informed unsupervised deep learning for adaptive holographic imaging

We provied pytorch(python) and matlab implementations of **Physics-informed unsupervised deep learning for adaptive holographic imaging**. This code was written by **Chanseok Lee**.

# Overview
In optical imaging, image retrieval process often relies on inverse mapping between a measurement domain and an object domain. Deep learning methods have recently been shown to provide a fast and accurate framework to learn the inverse mapping. However, because the learning process solely depends on the statistical distribution of matched reference data, the reliability is in general compromised in practical imaging configurations where physical perturbations exist in various forms, such as mechanical movement and optical fluctuation. Here, in a holographic imaging scheme, we present a physics-informed unsupervised deep learning method that incorporates a parameterized forward physics model to render the adaptability in the inverse mapping even without any matched reference data for training. We show that both the morphology and the range of objects can be reconstructed under highly perturbative configurations where the object-to-sensor distance is set beyond the range of a given training data set. To prove the reliability of the proposed method in practical biomedical applications, we further demonstrate holographic imaging of red blood cells flowing in a cluster and diverse types of tissue sections presented without any ground truth data. Our results suggest that the physics-informed unsupervised approach effectively extends the adaptability of deep learning methods, and therefore, has great potential for solving a wide range of inverse problems in optical imaging techniques.  
<p align = "center">
<img src="/image/simple_scheme.png" width="900" height="350">
</p>

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
- scipy

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
Test model with **MNIST** dataset. Complex amplitude of the sample reconstructed from single hologram intensity measurement can be compared with ground truth. User can train the network from scartch or download network parameters from [here](https://drive.google.com/drive/folders/1Y6R8plKylzHNT4wkBEA4GeOreY9id1xm?usp=sharing.). After downloading (mnist_complex_amplitude_5depth, mnist_amplitude_5depth or mnist_phase_5depth), put downloaded folder to **./model_parameters** folder.

### Test with MNIST
data_name: mnist
```
python test.py --data_name mnist --num_depth 5 --result_root RESULT_PATH --mode complex_amplitude
```

## Reproduce
Here, user can reproduce the reported results from Fig 2 to Fig 5 using below commands.  
Also, trained parameters used in this study can be downloaded from [here](https://drive.google.com/drive/folders/1Y6R8plKylzHNT4wkBEA4GeOreY9id1xm?usp=sharing.). Download folders and put them to **./model_parameters** folder.  

### Demonstration of simultaneous reconstruction of complex amplitude and object distance
```
python result./result_fig2.py
```
Inference time(CPU): 5s

### Demonstration of adaptive holographic imaging
```
python result./result_fig3.py
```
Inference time(CPU): 15s

### Demonstration of holographic imaging of RBCs in a dynamic environment
```
python result./result_fig4.py
```
Inference time(CPU): 5s

### Holographic imaging of histology slides without ground truth
```
python result./result_fig5.py
```
Inference time(CPU): 15s
