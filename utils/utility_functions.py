import math
from skimage import io, color
from skimage.transform import resize
import numpy as np


def initialize_cluster_centers_np(k, img_h, img_w):
  # Params needed for calculation of initial cluster center indices
  r = int(math.sqrt(k))
  c = k // r
  ratio_width =  (img_w - 10) // c
  ratio_height = (img_h - 10) // r
  row_offset = 10
  col_offset = 15

  # Determination of superpixel indices
  row_positions = ratio_height * np.arange(r, dtype=np.int32) + (row_offset * np.ones(r))
  col_positions = ratio_width * np.arange(c, dtype=np.int32) + (col_offset * np.ones(c))
  row_indices = np.broadcast_to(row_positions, (len(col_positions), len(row_positions)))
  col_indices = np.broadcast_to(col_positions, (len(row_positions), len(col_positions)))
  superpixel_indices = np.stack((row_indices, col_indices.T), axis=-1).reshape(-1, 2)

  # If k is not a perfect square, some superpixels arent generated, so we randomly generate them to get k superpixels
  num_generated = len(superpixel_indices)
  if num_generated < k:
        remaining = k - num_generated
        y_indices = np.random.randint(20, img_h - 20, remaining)
        x_indices = np.random.randint(20, img_w - 20, remaining)
        leftover_superpixels = np.stack((y_indices, x_indices), axis=-1).reshape(-1, 2)
        superpixel_indices = np.vstack((superpixel_indices, leftover_superpixels))

  return superpixel_indices


# calculates gradient considering all three color channels
def calc_gradient_map(img, img_h, img_w):
  grad_map = np.zeros((img_h , img_w), dtype=np.float32)
  for i in range(3):
    map = np.zeros((img_h + 2, img_w + 2), dtype=np.float32)
    np.copyto(map[1:-1, 1:-1], img[i])
    diff_map_x = map[:, 2:] - map[:, :-2]
    grad_map_x = diff_map_x[0: - 2, :] + diff_map_x[2: , :]
    grad_map = grad_map + grad_map_x

  return grad_map


# Function which reassign the cluster center to the pixel having the lowest gradient
def reassign_cluster_center_acc_to_grad_np(clusters, grad_map):
  y_indices = []
  x_indices = []

  # Changes the cluster index to the one having min grad value in a 3 x 3 region around it
  for cluster in clusters:
    idx_y, idx_x = int(cluster[0]), int(cluster[1])
    curr_window = grad_map[idx_y - 1 : idx_y + 2, idx_x - 1: idx_x + 2]
    min_element = np.argmin(curr_window)
    y_min = (min_element // 3) + idx_y
    x_min = (min_element % 3) + idx_x

    y_indices.append(y_min)
    x_indices.append(x_min)

  return y_indices, x_indices


# function to convert LAB images back to RGB and save it
def lab2rgb(path, lab_arr):
    rgb_arr = color.lab2rgb(lab_arr)
    rgb_arr = (rgb_arr * 255).astype(np.uint8)
    io.imsave(path, rgb_arr)
