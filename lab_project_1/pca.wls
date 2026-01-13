#!/usr/bin/env wolframscript
(* ::Package:: *)

mnist = ResourceData["MNIST", "TrainingData"];
group[i_] := group[i] = Select[mnist, #[[2]] == i &][[All, 1]];
nums = {4, 7, 8};
data = Join @@ group /@ nums;
vec[img_] := Flatten[ImageData[img]]
meanVec[set_] := Total[vec /@ set]/Length[set]
meanVecP[set_] := Total[Parallelize[vec /@ set]]/Length[set]
S[set_] := Block[{mean = meanVec[set], mat = ConstantArray[0, {784, 784}]},
Do[Block[{hey = vec[img] - mean}, mat += KroneckerProduct[hey, hey]], {img, set}];
mat/Length[set]]
SP[set_] := Block[{mean = meanVecP[set], mat},
ParallelEvaluate[mat = ConstantArray[0, {784, 784}]];
ParallelDo[Block[{hey = vec[img] - mean}, mat += KroneckerProduct[hey, hey]], {img, set}, Method -> "CoarsestGrained"];
Total[ParallelEvaluate[mat]]/Length[set]]
AbsoluteTiming[mat = SP[data];]
(*nums = Range[0, 9];
data = Join @@ group /@ nums;
AbsoluteTiming[mat = SP[data];]*)
pca[mat_, n_] := Eigenvectors[mat, n]
eigs = Eigenvalues[mat];
runningTotal = Accumulate[eigs];
SelectFirst[runningTotal, # >= 0.98 runningTotal[[-1]] & -> "Index"]
totalDistortion[A_, set_] := Block[{mean = meanVec[set], proj = Transpose[A] . A}, Sum[Block[{hey = vec[img] - mean}, Norm[proj . hey - hey]^2], {img, set}]]
dim = 10;
A = pca[mat, dim];
AbsoluteTiming[totalDistortion[A, data]]
Table[totalDistortion[pca[mat, dim], data], {dim, {2, 10, 50, 100, 200, 300}}]
A2 = pca[mat, 2];
ListPlot[Transpose[A2 . Transpose[vec /@ group[#]]] & /@ nums]
classMean[i_] := classMean[i] = meanVec[group[i]]
cov[i_] := cov[i] = SP[group[i]];
totalscatter = Sum[cov[i], {i, nums}];
betweenscatter = Block[{mean = meanVec[data]}, Sum[Block[{hey = classMean[i] - mean}, Length[group[i]] Outer[Times, hey, hey]], {i, nums}]];
ldaMat[\[Lambda]_] := Inverse[totalscatter + \[Lambda] IdentityMatrix[784]] . betweenscatter
lda[n_] := Eigenvectors[ldaMat[10^-4], n]
B2 = Chop[lda[2]];
ListPlot[Transpose[B2 . Transpose[vec /@ group[#]]] & /@ nums, ImageSize -> Full]
B3 = Chop[lda[3]];
ListPointPlot3D[Transpose[B3 . Transpose[vec /@ group[#]]] & /@ nums, ImageSize -> Full]



