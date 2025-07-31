# HRNet Pose Estimation Model Adapter

## Introduction

This repository provides a model integration between HRNet for human pose estimation and [Dataloop](https://dataloop.ai/).

HRNet (High-Resolution Network) is a state-of-the-art deep neural network for human pose estimation that maintains high-resolution representations throughout the entire process. It produces accurate keypoint detection for single and multiple persons by connecting high-to-low resolution convolutions in parallel and repeatedly exchanging information across resolutions.

## Available Model Weights

You can choose from three different HRNet weight configurations to balance between accuracy and performance:

### 1. HRNet-W48 (384×288) - High Accuracy
- **Configuration**: `hrnet_w48_384x288` 
- **File**: `pose_hrnet_w48_384x288.pth`
- **Parameters**: 48 channels, 384×288 input resolution
- **Description**: Best accuracy for pose estimation with higher computational requirements
- **Use Case**: When accuracy is the top priority and computational resources are available
- **Download**: [Google Drive](https://drive.google.com/uc?id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS)

### 2. HRNet-W32 (256×192) - Balanced Performance  
- **Configuration**: `hrnet_w32_256x192`
- **File**: `pose_hrnet_w32_256x192.pth`
- **Parameters**: 32 channels, 256×192 input resolution
- **Description**: Good balance between accuracy and speed
- **Use Case**: General purpose pose estimation with moderate computational requirements
- **Download**: [Google Drive](https://drive.google.com/uc?id=1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38)

### 3. HRNet-W32 (256×256) - Square Input Format
- **Configuration**: `hrnet_w32_256x256`
- **File**: `pose_hrnet_w32_256x256.pth` 
- **Parameters**: 32 channels, 256×256 input resolution
- **Description**: Square input format model for specific use cases
- **Use Case**: When square input images are preferred or required
- **Download**: [Google Drive](https://drive.google.com/uc?id=1_wn2ifmoQprBrFvUCDedjPON4Y6jsN-v)

## Setup Requirements

For the HRNet model to work properly with Dataloop, you must configure two essential components:

### 1. Required Ontology Configuration

The model expects exactly 17 keypoint labels in the COCO format.

#### Required Keypoint Labels

Your ontology must include these exact labels for the 17 COCO body keypoints:

1. **nose** - Nose point
2. **left_eye** - Left eye point
3. **right_eye** - Right eye point  
4. **left_ear** - Left ear point
5. **right_ear** - Right ear point
6. **left_shoulder** - Left shoulder joint
7. **right_shoulder** - Right shoulder joint
8. **left_elbow** - Left elbow joint
9. **right_elbow** - Right elbow joint
10. **left_wrist** - Left wrist/hand point
11. **right_wrist** - Right wrist/hand point
12. **left_hip** - Left hip joint
13. **right_hip** - Right hip joint
14. **left_knee** - Left knee joint
15. **right_knee** - Right knee joint
16. **left_ankle** - Left ankle/foot point
17. **right_ankle** - Right ankle/foot point

#### Setting Up Your Ontology

1. **Use the Example**: You can find a complete ontology example in `assets/ontology_example.json` that includes all required keypoints with appropriate colors and labels.

2. **Create in Dataloop**: Import this ontology structure into your Dataloop project through the platform interface or SDK.

3. **Recipe Configuration**: Use the recipe example in `assets/recipe_example.json` as a template, which shows how the ontology should be configured for pose estimation tasks.

**Important**: The keypoint labels must match exactly as shown above (case-sensitive) for the model to function correctly. The model maps its output keypoints to these specific COCO format label names.

### 2. Required Annotation Templates

You must create annotation templates that define the keypoint structure for pose estimation annotations.

#### Template Requirements

- **Template Name**: Must be named exactly **"person"**
- **Template Type**: Point collection template with 17 keypoints
- **Point Labels**: Each point must match the ontology labels exactly
- **Point Order**: Must follow the COCO keypoint sequence listed below

#### Creating the Annotation Template in Dataloop

Follow these step-by-step instructions:

1. **Navigate to Dataset Recipe**:
   - Go to your Dataloop project
   - Open your dataset
   - Click on the **"Recipe"** tab
   - Select the **"Instructions"** tab within the Recipe section

2. **Create New Annotation Template**:
   - In the Instructions tab, look for the "Annotation Templates" section
   - Click **"Create New Template"** or **"Add Template"**
   - Set the template name to exactly **"person"**
   - Choose **"Point Collection"** as the template type

3. **Add Keypoints to Template**:
   Add 17 points to the template with these labels in exact COCO order:
   1. `nose`
   2. `left_eye`
   3. `right_eye`
   4. `left_ear`
   5. `right_ear`
   6. `left_shoulder`
   7. `right_shoulder`
   8. `left_elbow`
   9. `right_elbow`
   10. `left_wrist`
   11. `right_wrist`
   12. `left_hip`
   13. `right_hip`
   14. `left_knee`
   15. `right_knee`
   16. `left_ankle`
   17. `right_ankle`

4. **Configure Point Positions**:
   - Arrange the points in a human pose formation
   - You can use the `assets/recipe_example.json` file as reference for default point positions
   - Save the template once all points are configured

**Critical Notes**:
- The template name "person" is required for the model to recognize annotations
- Point labels must match the ontology labels exactly (case-sensitive)
- The order of points follows the standard COCO keypoint format
- All 17 keypoints must be included in the template

## Model Configuration

### Configuration Options

When creating or configuring your HRNet model, you can specify the following parameters:

```json
{
  "weight_config": "hrnet_w48_384x288",  // Choose from available weight configurations
  "hrnet_joints_set": "coco",           // Joint set format (default: "coco")
  "single_person": false,               // Single person detection mode
  "yolo_version": "v5",                // YOLO version for person detection
  "use_tiny_yolo": false,              // Use lightweight YOLO model
  "disable_tracking": false,           // Disable person tracking in videos
  "max_batch_size": 16,                // Maximum batch size for inference
  "device": null,                      // Device for inference (null = auto-detect)
  "enable_tensorrt": false,            // Enable TensorRT optimization
  "bounding_boxes": true               // Adding bounding box around the annotated object
}
```

### Weight Configuration Selection

The `weight_config` parameter determines which HRNet model weights to use:

- `"hrnet_w48_384x288"` (default) - High accuracy model
- `"hrnet_w32_256x192"` - Balanced model  
- `"hrnet_w32_256x256"` - Square input model

### Automatic Weight Download

The adapter automatically downloads the appropriate weight file based on your configuration:

1. **Automatic Detection**: When the model loads, it checks which weight file is needed
2. **Smart Download**: Only downloads the specific weight file for your configuration
3. **Caching**: Downloaded weights are cached locally to avoid re-downloading
4. **Error Handling**: Provides clear error messages if download fails

## Performance Comparison

| Model Configuration | Accuracy | Speed | Memory Usage | Best Use Case |
|-------------------|----------|-------|--------------|---------------|
| HRNet-W48 (384×288) | Highest | Slower | High | Production accuracy-critical applications |
| HRNet-W32 (256×192) | Good | Faster | Medium | General purpose pose estimation |
| HRNet-W32 (256×256) | Good | Faster | Medium | Square image inputs, mobile applications |

## System Requirements

- dtlpy
- torch>=1.9.0
- torchvision>=0.10.0
- opencv-python>=4.5.0
- numpy>=1.21.0
- ultralytics>=8.3.152
- matplotlib>=3.3.0
- ffmpeg-python>=0.2.0
- munkres>=1.1.4
- scipy>=1.7.0
- Pillow>=8.3.0
- gdown>=4.6.0
- An account in the [Dataloop platform](https://console.dataloop.ai/)

## Sources and Further Reading

- [HRNet Documentation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
- [Simple-HRNet Implementation](https://github.com/stefanopini/simple-HRNet)
- [COCO Keypoint Detection](https://cocodataset.org/#keypoints-2020)

## Acknowledgements

The original models paper and codebase can be found here:
- HRNet paper on [arXiv](https://arxiv.org/abs/1902.09212) and codebase on [GitHub](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).
- Simple-HRNet wrapper on [GitHub](https://github.com/stefanopini/simple-HRNet).

We appreciate their efforts in advancing the field and making their work accessible to the broader community.