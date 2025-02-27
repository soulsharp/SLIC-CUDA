#include<cmath>
#include <cfloat>

// Gets the vector sum of inputVector where each row is a separate vector and also an average(refer to code for exact form of avg)
extern "C" __global__ void fusedMultiVectorSumAndAverage(float* inputVector, float* outputRow, int numRows, int numCols){

  __shared__ float accumulateSum[1024];
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  // Memory bounds check
  if(tid >= numCols) return;

  // Number of threads needed for average
  int numThreadsForAveraging = numCols / 6;
  accumulateSum[tid] = 0.0f;

  for(int i = 0; i < numRows; ++i){

    // Accesses the start of a new row each iteration
    int offset = numCols * i;

    // Accumulates sum in the first row
    accumulateSum[tid]  += inputVector[offset + tid];

  }

  __syncthreads();

  int idx = tid * 6;

  // Division takes place only for elements whose count is greater than 0
  if(tid < numThreadsForAveraging && accumulateSum[idx + 5] > 0.0){
    outputRow[idx + 0] = accumulateSum[idx + 0] / accumulateSum[idx + 5];
    outputRow[idx + 1] = accumulateSum[idx + 1] / accumulateSum[idx + 5];
    outputRow[idx + 2] = accumulateSum[idx + 2] / accumulateSum[idx + 5];
    outputRow[idx + 3] = accumulateSum[idx + 3] / accumulateSum[idx + 5];
    outputRow[idx + 4] = accumulateSum[idx + 4] / accumulateSum[idx + 5];
    outputRow[idx + 5] = 0.0f;
  }
}

extern "C" __global__ void assignClusterCenters(float* L, float* A, float* B, float* clusterPtr,
                                  int* labelMap, float* output, int imageWidth, int imageHeight,
                                  int numSuperpixels, int M, int S){



  extern __shared__ float sharedMem[];

  int tid = threadIdx.x;
  int global_tid = tid + blockIdx.x * blockDim.x;

  // Memory bounds check
  if(global_tid >= imageWidth * imageHeight) return;
  int Y = global_tid % imageWidth;
  int X = global_tid / imageWidth;
  int lastBlockId = (imageWidth * imageHeight + blockDim.x - 1) / blockDim.x;

  // Total number of values to load in shared memory
  int numValues = numSuperpixels * 6;

  // clusterInfo is read-only, updateBuffer accumulates sums to aggregate them later in global memory
  float* clusterInfo = sharedMem;
  float* updateBuffer = sharedMem + numValues;

  // Load clusterPtr to shared memory
  if(blockIdx.x == lastBlockId - 1){
    int numLegalThreads = imageWidth * imageHeight - (lastBlockId - 1) * blockDim.x;
    for(int i = tid; i < numValues; i+=numLegalThreads){
      clusterInfo[i] = clusterPtr[i];
      updateBuffer[i] = 0.0;
    }
   }
  else{
    if (tid < numValues){
      clusterInfo[tid] = clusterPtr[tid];
      updateBuffer[tid] = 0.0;
    }
   }


  __syncthreads();

  float minDist = FLT_MAX;
  int minClusterIdx = 0;
  float eucDistFactor = static_cast<float>(M) / static_cast<float>(S);

  // computes distance to relevant cluster centers and assign pixel to the nearest one
  for(int i = 0; i < numSuperpixels; ++i){
    float cL = clusterInfo[i * 6 + 0];
    float cA = clusterInfo[i * 6 + 1];
    float cB = clusterInfo[i * 6 + 2];
    float cX = clusterInfo[i * 6 + 3];
    float cY = clusterInfo[i * 6 + 4];

    float absDistX = fabsf(cX - X);
    float absDistY = fabsf(cY - Y);

    // Only cluster centers in a 2S x 2S window around the pixel can influence it
    if(absDistX <= 2 * S && absDistY <= 2 * S){

      // Euclidean distance
      float eucDist = sqrtf(absDistX * absDistX + absDistY * absDistY);

      // LAB color distance
      float labDist = sqrtf((cL - L[global_tid]) * (cL - L[global_tid]) +
                      (cA - A[global_tid]) * (cA - A[global_tid]) +
                      (cB - B[global_tid]) * (cB - B[global_tid]));

      // Total distance
      float totalDist = eucDistFactor * eucDist + labDist;

      // Gets the nearest cluster's ID
      if(totalDist < minDist){
        minClusterIdx = i + 1;
        minDist = totalDist;
    }
 }

}

  // Assigns labelMap location corresponding to the tid, the min cluster ID
  labelMap[global_tid] = minClusterIdx;

  // -1 is needed because minClusterIdx goes from 1 -> numSuperpixels + 1
  int minClusterIdxStartShared = (minClusterIdx - 1) * 6 ;

  // Add values to the updateBuffer for averaging later
  atomicAdd(&updateBuffer[minClusterIdxStartShared + 0], L[global_tid]);
  atomicAdd(&updateBuffer[minClusterIdxStartShared + 1], A[global_tid]);
  atomicAdd(&updateBuffer[minClusterIdxStartShared + 2], B[global_tid]);
  atomicAdd(&updateBuffer[minClusterIdxStartShared + 3], X);
  atomicAdd(&updateBuffer[minClusterIdxStartShared + 4], Y);
  atomicAdd(&updateBuffer[minClusterIdxStartShared + 5], 1.0);

  __syncthreads();

  // Writes updateBuffer(shared memory) to the correct location at output buffer(global memory)
  int outputRow = blockIdx.x;
  int startIdxOutputBuffer = outputRow * numValues;

  if(blockIdx.x == lastBlockId - 1){
    int numLegalThreads = imageWidth * imageHeight - (lastBlockId - 1) * blockDim.x;

    for(int i = tid; i < numValues; i+=numLegalThreads){
      output[startIdxOutputBuffer + i] = updateBuffer[i];
    }
  }

  else{

    if (tid < numValues) {
          output[startIdxOutputBuffer + tid] = updateBuffer[tid];
      }
  }

}

extern "C" __global__ void averageColorCluster(float* L, float* A, float* B, float* clusterPtr,
                                  int* labelMap, int imageWidth, int imageHeight){

  int tid = threadIdx.x;
  int global_tid = tid + blockIdx.x * blockDim.x;

  // Memory bounds check
  if(global_tid >= imageWidth * imageHeight) return;

  int label = labelMap[global_tid] - 1;

  L[global_tid] = clusterPtr[label * 6 + 0];
  A[global_tid] = clusterPtr[label * 6 + 1];
  B[global_tid] = clusterPtr[label * 6 + 2];
  }

