Note: The Matlab codes will only work after downloading MSR-ACTION3D, unable to upload the entire dataset.
Please Download the MSR-ACTION3D DATASET FROM THE LINK BELOW
https://www.uow.edu.au/~jz960/datasets/MSRAction3D.html (requires password to open the file)

## This Matlab Source code is based on the paper Titled,
## "Online action recognition from RGB-D cameras based on reduced basis decomposition"
## written by Muniandi Arunraj, Andy Srinivasan, A. Vimala Juliet
## [Paper Link:](https://link.springer.com/article/10.1007/s11554-018-0778-8)
## [Read only Link:](https://rdcu.be/NyqK )

## (used Bicubic Interplations (resize_feature1.m) and Lanczos 3rd order Interpolations (resize_feature2.m)   )

# 1.AS_Imagereconstruction.m, 

 AS_Imagereconstruction.m - used for reproducing the image results in the paper, 
 particularly reconstructing the images under various proportions
 (10%, 20%, 30%, 40%, 50% upto 100%)
 using 
 Reduced Basis Decomposition(RBD),
 Principal Component Analysis(PCA), 
 Singular Value Decomposition(SVD)
 
 
# 2. ASFlopcounts.m   

 Note: FLOPS require function variables(RBD,Pro-CRC,Pro-Max,Eigenface_f,L2CRC) 
 to be stored in the workspace(results folder), time won't give you the correct 
 evaluation due to optimization problems and memory management issues within 
 MATLAB. However the time difference between RBD and PCA can be noticed when run on 
 either MAC(As per RBD Author)/Linux(As per the current paper).  
 
 ASFlopcounts.m - used for comparing the FLOPS between 
 1.RBD vs PCA 
 2.Pro-CRC vs L2-CRC
 
 
# 3."AS1_crossfixedbicubic","AS2_crossfixedbicubic","AS3_crossfixedbicubic",
# "AS1_crossfixedlanczos","AS2_crossfixedlanczos","AS3_crossfixedlanczos",
(Note:- Although these tests were followed in most RGB-D related papers, its not 
a good test to compare classification effectiveness with previous papers)


# 4."AS1_LOSObicubic","AS2_LOSObicubic","AS3_LOSObicubic",
# "AS1_LOSOlanczos","AS2_LOSOlanczos","AS3_LOSOlanczos",
(Note:- Second best method, however each actionsets will have different settings 
for resizing images(front,side and top))


# 5."ASFullLOSO_bicubic","ASfull252combo_bicubic"
# "ASFullLOSO_lanczos","ASfull252combo_lanczos"
(Note1:- Best methods for producing close to real-time performance and all 
the actions involved in (AS1,AS2 and AS3) it follow the same settings 
for resizing images(front,side and top))
(Note2:- It contains exhaustive 252 Combinations of all subjects and LeaveOneSubjectOut
LOSO Tests) - Takes sometime to run 
 
# II. Major Functions used
# RBD.m 
(For computing reduced basis decomposition)
# Eigenface_f.m 
(For computing PCA)
# ProCRC 
(For finding the alpha coefficients of probabilistic Classification )
# ProMax 
(For finding the classified final label based on residual errors)
# L2CRC 
(Collaborative Representation classifier with Tikhonov Weighted Regularization)

# III. Supportive Functions used
# FLOPS 
(for computing the FLOPS, this may change based on hardware architecture and 
Operating systems)


