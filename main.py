if __name__ == "__main__":

  import argparse
  from utils.utility_functions import initialize_cluster_centers_np, calc_gradient_map, reassign_cluster_center_acc_to_grad_np, lab2rgb
  import os
  import math
  from skimage import io, color
  from skimage.transform import resize
  import numpy as np
  import pycuda.driver as cuda
  import pycuda.autoinit
  from pycuda.compiler import SourceModule

  parser = argparse.ArgumentParser()
  
  # Please put input image in the results/ images dir
  parser.add_argument("--image_path", type=str, help="Path of the image to be SLIC-ed")
  
  # Number of superpixels, number of iterations and normalizing factor to be used in the image
  parser.add_argument("--num_superpixels", type=int, help="Number of superpixels in the image", default=150)
  parser.add_argument("--num_iterations", type=int, help="Number of iterations to run SLIC for", default=10)
  parser.add_argument("--M", type=int, help="Normalizing factor used in distance calculations", default=10)

  args = parser.parse_args()

  # Saves CLI arguments
  num_superpixels = args.num_superpixels
  num_iterations = args.num_iterations
  M = args.M
  img_path = args.image_path

  # Reads the input RGB image and resizes it 
  rgb = io.imread(img_path)
  rgb = resize(rgb, (400, 400))

  # Converts RGB to LAB
  img = color.rgb2lab(rgb)

  image = np.transpose(img, (2, 0, 1))
  image_height = img.shape[0]
  image_width = img.shape[1]

  # Kernel launch and superpixel config
  N = image_width * image_height
  threadsPerBlock = 1024
  numBlocks = (N + threadsPerBlock - 1) // threadsPerBlock
  num_superpixels = 170
  num_rows_output = np.int32(numBlocks)
  num_cols_output = np.int32(num_superpixels * 6)
  S = int(math.sqrt(N /num_superpixels))

  # Gets the initial cluster center
  clusters = initialize_cluster_centers_np(num_superpixels, image_height, image_width)

  # Gets the gradient map
  grad_map = calc_gradient_map(image, image_height, image_width)

  # Gets the X, Y coordinates of the cluster centers after perturbing them based on gradient
  rows, cols = reassign_cluster_center_acc_to_grad_np(clusters, grad_map)

  L = image[0][rows, cols]
  A = image[1][rows, cols]
  B = image[2][rows, cols]
  counts = np.zeros_like(L , np.float32)

  # Constructs superpixels in the form that is expected by assignClusterCenters kernel
  cluster_array = np.stack((L, A, B, rows, cols, counts), axis=-1).ravel()
    
  # Arrays on host
  L_array = img[:, :, 0].ravel().astype(np.float32)
  A_array = img[:, :, 1].ravel().astype(np.float32)
  B_array = img[:, :, 2].ravel().astype(np.float32)

  # Holds the pixel labels
  label_array = np.zeros(image_height * image_width).astype(np.int32).ravel()

  # Holds the intermediate outputs of the shape (num_superpixels, 6) which contains respective contributions to cluster centers
  output_array = np.zeros((num_rows_output, num_cols_output), dtype=np.float32).ravel()
  cluster_array = cluster_array.astype(np.float32)
  label_array_final = np.zeros_like(label_array).astype(np.int32)
  L = np.zeros((image_height, image_width)).astype(np.float32)
  A = np.zeros((image_height, image_width)).astype(np.float32)
  B = np.zeros((image_height, image_width)).astype(np.float32)
  updated_cluster_array = np.zeros(num_cols_output).astype(np.float32)

  # Gets sizes needed for allocations on the GPU
  size_LAB = L_array.nbytes
  size_cluster_array = cluster_array.nbytes
  size_label_array = label_array.nbytes
  size_output_array = output_array.nbytes

  # Allocates arrays on the GPU
  d_l = cuda.mem_alloc(size_LAB)
  d_a = cuda.mem_alloc(size_LAB)
  d_b = cuda.mem_alloc(size_LAB)
  d_cluster = cuda.mem_alloc(size_cluster_array)
  d_label = cuda.mem_alloc(size_label_array)
  d_output = cuda.mem_alloc(size_output_array)
  d_input_vector = cuda.mem_alloc(size_output_array)
  d_cluster_mean = cuda.mem_alloc(size_cluster_array)


  # Copies arrays from CPU to GPU
  cuda.memcpy_htod(d_l, L_array)
  cuda.memcpy_htod(d_a, A_array)
  cuda.memcpy_htod(d_b, B_array)
  cuda.memcpy_htod(d_cluster, cluster_array)
  cuda.memcpy_htod(d_label, label_array)
  cuda.memcpy_htod(d_output, output_array)
  cuda.memcpy_htod(d_input_vector, output_array)
  cuda.memcpy_htod(d_cluster_mean, updated_cluster_array)
  
  # Reads CUDA functions from slic_kernels.cu
  kernel_path = os.path.join(os.path.dirname(__file__), "kernels", "slic_kernels.cu")
  with open(kernel_path, "r") as f:
      kernels = f.read()

  kernel = SourceModule(kernels)
 
  # Gets functions
  assign_cluster_fn = kernel.get_function("assignClusterCenters")
  update_cluster_fn = kernel.get_function("fusedMultiVectorSumAndAverage")
  average_color_fn = kernel.get_function("averageColorCluster")

  size_smem = 2 * size_cluster_array

  # Performs the SLIC algo iterations 10 times
  for i in range(num_iterations):
    assign_cluster_fn(d_l, d_a, d_b, d_cluster, d_label, d_output,
            np.int32(image_width), np.int32(image_height), np.int32(num_superpixels), np.int32(M), np.int32(S),
            block=(threadsPerBlock, 1, 1),
            grid = (numBlocks, 1), shared=size_smem)

    cuda.memcpy_dtoh(output_array, d_output)

    update_cluster_fn(d_output, d_cluster, num_rows_output, num_cols_output,
            block=(threadsPerBlock, 1, 1),
            grid = (1, 1))

  # Averages the colors of superpixels
  average_color_fn(d_l, d_a, d_b, d_cluster, d_label, np.int32(image_height), np.int32(image_width),
                  block=(threadsPerBlock, 1, 1),
                  grid = (numBlocks, 1))
  cuda.Context.synchronize()

  # Moves arrays back to host from device for saving the results
  cuda.memcpy_dtoh(label_array_final, d_label)
  cuda.memcpy_dtoh(cluster_array, d_cluster)
  cuda.memcpy_dtoh(L, d_l)
  cuda.memcpy_dtoh(A, d_a)
  cuda.memcpy_dtoh(B, d_b)
  lab_img = np.stack([L, A, B], axis=-1)
  
  suffix = img_path.split('\\')[-1]
  suffix = "SLIC_Output_" + suffix

  # Save path
  save_path = os.path.join(os.getcwd(), "results", "slic_outputs", suffix)

  # Ensures the directory exists
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  lab2rgb(save_path, lab_img)
