# Robust Image Watermarking using SVD and Dither Quantization

This project implements and analyzes a robust digital image watermarking method based on **Singular Value Decomposition (SVD)** and **Dither Quantization**.  
The goal is to embed a binary watermark into a grayscale host image while preserving visual quality and improving robustness against common image processing attacks.

The project was developed as part of a university seminar on **Steganography, Matrix Methods, and Tensor Methods**.

## Project Overview

Digital watermarking is used to protect multimedia content by embedding ownership information, such as a logo or binary watermark, into a host image.  
This project focuses on a transform-domain watermarking approach, where watermark bits are embedded using SVD-based image decomposition.

Unlike simple spatial-domain techniques, transform-domain methods are generally more robust against compression, noise, filtering, cropping, and other attacks.

## Main Idea

The host image is divided into four subimages. The watermark is embedded only into selected diagonal subimages in order to preserve the visual quality of the final watermarked image.

Watermark data is embedded in two different SVD components:

- the **D matrix**, using the largest singular values and dither quantization
- the **U matrix**, using relationships between selected matrix coefficients

This provides two possible extraction paths. If watermark extraction from one component is less successful after an attack, the second component may still preserve useful watermark information.

## Methodology

The watermarking process consists of the following steps:

1. Convert the host image to grayscale.
2. Resize and binarize the watermark image.
3. Partition the host image into four subimages.
4. Apply SVD to selected image blocks.
5. Embed watermark bits into:
   - singular values of the D matrix
   - coefficients of the U matrix
6. Reconstruct the watermarked image.
7. Apply different attacks to test robustness.
8. Extract the watermark from both D and U matrices.
9. Evaluate the quality using standard performance metrics.

## Techniques Used

- Singular Value Decomposition (SVD)
- Dither Quantization
- Image partitioning
- Binary watermark embedding
- Watermark extraction
- Robustness testing under image attacks
- Performance evaluation using PSNR, BER, and NC

## Performance Metrics

The following metrics were used to evaluate the method:

### PSNR — Peak Signal-to-Noise Ratio

Measures the visual quality of the watermarked or attacked image.  
Higher PSNR usually means better image quality.

### BER — Bit Error Rate

Measures the percentage of incorrectly extracted watermark bits.  
Lower BER means better extraction accuracy.

### NC — Normalized Correlation

Measures similarity between the original watermark and the extracted watermark.  
Higher NC means better watermark recovery.

## Tested Attacks

The method was tested against multiple image processing attacks, including:

- JPEG compression
- Gaussian noise
- Rotation
- Resizing
- Median filtering
- Blur
- Salt and pepper noise
- Cropping
- Brightness adjustment
- Gamma correction
- Row-column blanking
- Row-column copying
- Bit-plane removal

## Experimental Results

The method showed strong robustness in several cases, especially for attacks such as:

- JPEG compression
- cropping
- gamma correction
- brightness adjustment
- bit-plane removal

For example, in many tests the extracted watermark from the U matrix achieved very high normalized correlation, especially after cropping, brightness changes, and gamma correction.

Some attacks, such as strong rotation and noise, caused larger degradation, which is expected because they significantly modify the image structure.

## Parameter Selection

Several parameter values were tested in order to balance image quality and watermark robustness.

The main parameters were:

- `T` — quantization step size
- `alpha` — margin used for embedding bits in the U matrix

Values that produced PSNR below 40 dB were discarded during parameter tuning.  
The final selected values were:

- `T = 60` for the Shelby image
- `T = 40` for the Lena image
- `alpha = 0.05`

These values provided a good trade-off between imperceptibility and extraction accuracy.

## Example Results

| Image | Attack | PSNR | D Matrix NC | D Matrix BER | U Matrix NC | U Matrix BER |
|---|---:|---:|---:|---:|---:|---:|
| Shelby | No attack | 41.645 dB | 1.000 | 0.000 | 0.999 | 0.001 |
| Shelby | JPEG Q70 | 38.594 dB | 1.000 | 0.000 | 0.783 | 0.336 |
| Shelby | Cropping 25% | 22.604 dB | 0.932 | 0.132 | 0.999 | 0.001 |
| Lena | No attack | 39.270 dB | 1.000 | 0.000 | 1.000 | 0.000 |
| Lena | JPEG Q70 | 38.481 dB | 1.000 | 0.000 | 0.998 | 0.004 |
| Lena | Gamma correction | 29.955 dB | 0.994 | 0.010 | 1.000 | 0.000 |

## Repository Structure

```text
.
├── images/                 # Host images and watermark images
├── results/                # Watermarked images, attacked images, extracted watermarks
├── src/                    # Source code
│   ├── embedding.py        # Watermark embedding functions
│   ├── extraction.py       # Watermark extraction functions
│   ├── attacks.py          # Image attack functions
│   └── metrics.py          # PSNR, BER, and NC calculations
├── notebooks/              # Experiments and parameter tuning
├── README.md
└── requirements.txt
