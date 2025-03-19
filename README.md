# SLIC-CUDA

Simple Linear Iterative Clustering (SLIC) is an image processing algorithm that segments an image into M superpixels—connected regions of pixels that share similar color, texture, or intensity properties. Superpixels help represent parts of larger objects and are particularly useful in image segmentation (a key task in computer vision). Instead of classifying individual pixels, a neural network can operate on superpixels, making segmentation more efficient. 

For a deeper dive into the SLIC algorithm, check out the [original paper](https://ieeexplore.ieee.org/document/6205760) or this [blog post](https://darshita1405.medium.com/superpixels-and-slic-6b2d8a6e4f08), which served as my introduction to the concept. The accompanying [repository](https://github.com/darshitajain/SLIC) served as the base CPU implementation.

This repository provides a CUDA implementation of SLIC. Given the algorithm’s computationally intensive yet highly parallelizable nature, it immensely benefits from a GPU implementation.

## Quickstart

1. Clone the repository:

```bash
git clone https://github.com/soulsharp/SLIC-CUDA
```

2. Install dependencies using pip:

```bash
pip install -r requirements.txt
```

3. Run the main file:

```bash
python main.py --image_path path_of_your_image
```

## Additional Arguments  
You can customize the execution with the following optional arguments:  

- `--num_superpixels <int>`: Specify the number of superpixels to generate (default: 150).The max number of superpixels supported by the algorithm is 170 at the moment. 
- `--M <int>`: Adjust the compactness factor, controlling the trade-off between color similarity and spatial proximity (default: 10).It is recommended not to set values too far from 10
- `--num_iterations <int>` : Specify the number of iterations to run the algorithm for (default:10).Convergence is observed after about 7-8 iterations however authors of the original paper recommend to run the algorithm for 10 iterations.

Example usage:  
```bash
python main.py --image_path path_of_your_image --num_superpixels 100 --M 8 --num_iterations 7
```

Results are saved to the results/slic_outputs folder.

## Caveats & Considerations
1. At present, the algorithm finds the superpixels, averages the color content of all pixels belonging to a superpixel, and assigns this color to all pixels inside it. This results in artistic oil painting-like images which are saved to the results/slic_outputs folder. If you wish to remove this functionality, you can remove or comment out the 'average_color_fn' and make a few additional changes to ensure functional correctness.

2. You need to have a CUDA-supported GPU in order to use this repository as it doesn't support a CPU version.

3. The images are resized to (400, 400) for best results by default and saved as such. If you want to resize the image back to its original dimensions, please insert the skimage.transform's resize function after the algorithm's completion in the main file. However, you then need to implement a logic to also transform the superpixel boundaries to a larger/ smaller image's dimensions.

4. At present, the maximum number of superpixels that can be used by the algorithm is limited by the shared memory of the GPU. For eg, on a Tesla T4 GPU(Google Colab), maximum number of superpixels cannot exceed 170.
A workaround in the future could be to use global memory instead of shared memory for loading initial cluster information(which is what the shared memory is used for), but this will undoubtedly make the algorithm slower.

5. At present, the last step of the algorithm prescribed in the paper, the 'enforce connectivity' step is missing from the implementation. The reason for this is two-fold: It is non-trivial to parallelize this step efficiently on a GPU and more importantly, there's no need for this, atleast from my experimentations, if you resize the image to something like (400, 400). However, I plan to add this functionality in the future which should remove the need to downsize larger images to (400, 400) to apply the algorithm. 

## Sample results

The base CPU-implementation mentioned earlier takes on an average of around 2 minutes(varies based on the image size) on colab's server grade CPUs. 
This implementaion, on a Tesla T4 GPU, takes less on an average of around 0.15 seconds to complete, achieving a massive speedup of the order of 1000x while maintaining functional correctness. The times in both the CPU and GPU implementations are only for the core algorithm and doesnt account for any pre-processing/ loading images/ allocating space on the GPU(It takes around 1.5 seconds for all this on a GPU).

Here are a couple of results:

### Original Images: 
![Monkey on a Ledge Original](results\images\manas-manikoth-t3BPY1BCAAc-unsplash.jpg)  

Photo by <a href="https://unsplash.com/@manasmanikoth?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Manas Manikoth</a> on <a href="https://unsplash.com/photos/a-monkey-sitting-on-a-ledge-t3BPY1BCAAc?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
      
![Scenery Original](results\images\philipp-neumann-DHijgFy-OkI-unsplash.jpg)
 Photo by <a href="https://unsplash.com/@philneumn?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Philipp Neumann</a> on <a href="https://unsplash.com/photos/green-trees-near-lake-under-white-clouds-and-blue-sky-during-daytime-DHijgFy-OkI?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
      
      
### Artistic Outputs:
![Monkey on a Ledge](results\slic_outputs\SLIC_Output_manas-manikoth-t3BPY1BCAAc-unsplash.jpg)

![Scenery](results\slic_outputs\SLIC_Output_philipp-neumann-DHijgFy-OkI-unsplash.jpg)

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)