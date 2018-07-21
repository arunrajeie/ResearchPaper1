#####################################################################
## This Matlab Source code is based on the paper Titled,
## "Online action recognition from RGB-D cameras based on reduced basis decomposition"
## written by Muniandi Arunraj, Andy Srinivasan, A. Vimala Juliet
## Paper Link: https://link.springer.com/article/10.1007/s11554-018-0778-8
## Read only Link: https://rdcu.be/NyqK

Note: The Matlab codes will only work after downloading MSR-ACTION3D, unable to upload the entire dataset.
Please Download the MSR-ACTION3D DATASET FROM THE LINK BELOW
https://www.uow.edu.au/~jz960/datasets/MSRAction3D.html (requires password to open the file)

Our Project contains two major files
1.MainTest1.m, 

 (MainTest1.m - used for reproducing the results in the paper, particularly 
 reconstructing the images using Reduced Basis Decomposition(RBD),
 Principal Component Analysis(PCA), Singular Value Decomposition(SVD))
 
 (MainTest1.m - used to finding the reduced dimensions and classification results
 for the same including Time, FLOPS and accuracy)
 
2.MainTest2.m 

 (In contains exhaustive 256 Combinations of all subjects test followed by
 LOSO Test, try to replace the already generated file with the original MSR-Action3D
 Dataset to get better results)

Major Functions used
RBD.m (For computing reduced basis decomposition)
Eigenface_f.m (For computing PCA)
ProCRC (For finding the alpha coefficients of probabilistic Classification )
ProMax (For finding the classified final label based on residual errors)
L2CRC (Collaborative Representation classifier with Tikhonov Weighted Regularization)

Supportive Functions used
FLOPS (for computing the FLOPS, this may change based on hardware architecture and 
Operating systems)
