#!/usr/bin/env wolframscript
(* ::Package:: *)

mnist = ResourceData["MNIST", "TrainingData"];
mnistTest = ResourceData["MNIST", "TestData"];
group[i_] := group[i] = Select[mnist, #[[2]] == i &][[All, 1]]
trainSize[i_] := trainSize[i] = Length[group[i]]
groupTest[i_] := groupTest[i] = Select[mnistTest, #[[2]] == i &][[All, 1]]
testSize[i_] := testSize[i] = Length[groupTest[i]]
mat[img_] := 1 - ImageData[img]


data = Developer`ToPackedArray[mat /@ mnist[[All, 1]]];
labels = mnist[[All, 2]];
testData = Developer`ToPackedArray[mat /@ mnistTest[[All, 1]]];
testLabels = mnistTest[[All, 2]];
nTrain = Length[labels];
nTest = Length[testLabels];
classes = DeleteDuplicates[labels];
nClasses = Length[classes];
pos = AssociationThread[classes -> Range[nClasses]];


maxpool[array_, spec_] := Block[{dims = Dimensions[array], rank, shape, pooledDims, reshapedArray},
rank = Length[dims];
If[rank != Length[spec], Return[$Failed]];
shape = DeleteCases[Riffle[dims/spec, {#, "pool"} & /@ spec], {1, "pool"}];
pooledDims = First /@ Position[shape, "pool"];
shape = DeleteCases[Flatten[shape], "pool"];
reshapedArray = ArrayReshape[array, shape];
ArrayReduce[Max, reshapedArray, pooledDims]]
maxpoolD[array_, spec_] := Block[{pooled = maxpool[array, spec], shift, aug, pre},
shift = Max[Abs[pooled]]/Sqrt[2];
aug = KroneckerProduct[pooled, ConstantArray[1., spec]] + shift;
pre = Chop[(array + shift)/aug - 1];
Normal[SparseArray[Position[pre, 0] -> 1, Dimensions[pre]]]]
arrayContract[t1_, inds1_, t2_, inds2_] := Block[{dims1 = Dimensions[t1], dims2 = Dimensions[t2], l = Length[inds1], tt1, tt2, contractedDims, dim1, dim2, transposePerm1, transposePerm2},
If[dims1[[inds1]] != dims2[[inds2]] || Length[inds2] != l, Return[$Failed]];
dim1 = Length[dims1];
dim2 = Length[dims2];
transposePerm1 = ConstantArray[0, dim1];
transposePerm2 = ConstantArray[0, dim2];
transposePerm1[[Join[Complement[Range[dim1], inds1], inds1]]] = Range[dim1];
transposePerm2[[Join[inds2, Complement[Range[dim2], inds2]]]] = Range[dim2];
tt1 = Transpose[t1, transposePerm1];
tt2 = Transpose[t2, transposePerm2];
dims1 = Dimensions[tt1];
dims2 = Dimensions[tt2];
contractedDims = Times @@ dims2[[;; l]];
dims1[[-l ;; ]] = Nothing;
dims2[[;; l]] = Nothing;
dims1 = Append[dims1, contractedDims];
dims2 = Prepend[dims2, contractedDims];
tt1 = ArrayReshape[tt1, dims1];
tt2 = ArrayReshape[tt2, dims2];
tt1 . tt2]


batchSize = 77;
lr = 10^-2;
epochs = 40;
dims = {{32, 3, 3}, {64, 3, 3}, {64, 3, 3}, {}, {7744}, {128}, {10}};


SeedRandom[0];
W[1] = RandomVariate[NormalDistribution[], dims[[1]]] * Sqrt[2. / Times @@ dims[[1, -2 ;;]]];
W[2] = RandomVariate[NormalDistribution[], Insert[dims[[2]], dims[[1, 1]], 2]] * Sqrt[2. / Times @@ Flatten[{dims[[1, 1]], dims[[2, -2 ;;]]}]];
W[3] = RandomVariate[NormalDistribution[], Insert[dims[[3]], dims[[2, 1]], 2]] * Sqrt[2. / Times @@ Flatten[{dims[[2, 1]], dims[[3, -2 ;;]]}]];
W[5] = RandomVariate[NormalDistribution[], {dims[[5, 1]], 7744}] * Sqrt[2. / 7744];
b[5] = ConstantArray[0., dims[[5, 1]]];
W[6] = RandomVariate[NormalDistribution[], {dims[[6, 1]], dims[[5, 1]]}] * Sqrt[2. / dims[[5, 1]]];
b[6] = ConstantArray[0., dims[[6, 1]]];
W[7] = RandomVariate[NormalDistribution[], {dims[[7, 1]], dims[[6, 1]]}] * Sqrt[2. / dims[[6, 1]]];
b[7] = ConstantArray[0., dims[[7, 1]]];
W[1] = ArrayReshape[Transpose[W[1], {3, 1, 2}], {3 * 3, 32}];
W[2] = ArrayReshape[Transpose[W[2], {4, 3, 1, 2}], {3 * 3 * 32, 64}];
W[3] = ArrayReshape[Transpose[W[3], {4, 3, 1, 2}], {3 * 3 * 64, 64}];
Windices = {1, 2, 3, 5, 6, 7};
bindices = {5, 6, 7};


dist[img_] := Block[{A, Z, y}, A[1] = ArrayReshape[Partition[img, {3, 3}, {1, 1}], {26, 26, 3 * 3}] . W[1];
Z[1] = Ramp[A[1]];
A[2] = ArrayReshape[Partition[Z[1], {3, 3}, {1, 1}], {24, 24, 3 * 3 * 32}] . W[2];
Z[2] = Ramp[A[2]];
A[3] = ArrayReshape[Partition[Z[2], {3, 3}, {1, 1}], {22, 22, 3 * 3 * 64}] . W[3];
Z[3] = Ramp[A[3]];
A[4] = Transpose[maxpool[Z[3], {2, 2, 1}], {2, 3, 1}];
Z[4] = Flatten[A[4]];
A[5] = W[5] . Z[4] + b[5];
Z[5] = Ramp[A[5]];
A[6] = W[6] . Z[5] + b[6];
Z[6] = Ramp[A[6]];
A[7] = W[7] . Z[6] + b[7];
y = Exp[A[7]];
y = y / Total[y]]
pred[img_] := Block[{y = dist[img]}, classes[[FirstPosition[y, Max[y]][[1]]]]]
batchDist[block_, batchSize_ : 512(*batchSize*)] := If[Length[block] > batchSize, Block[{blocks = Partition[block, UpTo[batchSize]]}, Join @@ (Parallelize[batchDist[#, batchSize] & /@ blocks, Method -> "FinestGrained"])], Block[{A, Z, y}, Z[0] = block;
A[1]=Dot[ArrayReshape[Partition[#,{3,3},{1,1}],{26,26,3 3}]&/@Z[0],W[1]];
Z[1] = Ramp[A[1]];
A[2] = Dot[ArrayReshape[Partition[#, {3, 3}, {1, 1}], {24, 24, 3 * 3 * 32}] & /@ Z[1], W[2]];
Z[2] = Ramp[A[2]];
A[3] = Dot[ArrayReshape[Partition[#, {3, 3}, {1, 1}], {22, 22, 3 * 3 * 64}] & /@ Z[2], W[3]];
Z[3] = Ramp[A[3]];
A[4] = Transpose[maxpool[Z[3], {1, 2, 2, 1}], {1, 3, 4, 2}];
Z[4] = Flatten /@ A[4];
A[5] = # + b[5] & /@ (Z[4] . Transpose[W[5]]);
Z[5] = Ramp[A[5]];
A[6] = # + b[6] & /@ (Z[5] . Transpose[W[6]]);
Z[6] = Ramp[A[6]];
A[7] = # + b[7] & /@ (Z[6] . Transpose[W[7]]);
y = Exp[A[7]];
y = y / (Total /@ y)]]
batchPred[block_] := Block[{y = batchDist[block]}, classes[[FirstPosition[#, Max[#]][[1]]]] & /@ y]


AbsoluteTiming[preds = batchPred[data];]
matches[0] = MapThread[Equal, {preds, labels}];
correct[0] = Count[matches[0], True]
acc[0] = N[correct[0] / nTrain]


Monitor[Do[shuffle = Partition[RandomSample[Range[nTrain]], UpTo[batchSize]];
nBatches = Length[shuffle];
Do[batch = shuffle[[p]];
nBatch = Length[batch];
dataBatch = data[[batch]];
rVec = pos /@ labels[[batch]];
constructedW[3] = ArrayReshape[Transpose[Reverse[ArrayReshape[W[3], {9, 64, 64}]], {1, 3, 2}], {576, 64}];
constructedW[2] = ArrayReshape[Transpose[Reverse[ArrayReshape[W[2], {9, 32, 64}]], {1, 3, 2}], {576, 32}];
Block[{Z, A, y, label, labelVec, e, g3, g2, g1},
Z[0] = dataBatch;
A[1] = Dot[ArrayReshape[Partition[#, {3, 3}, {1, 1}], {26, 26, 3 * 3}] & /@ Z[0], W[1]];
Z[1] = Ramp[A[1]];
A[2] = Dot[ArrayReshape[Partition[#, {3, 3}, {1, 1}], {24, 24, 3 * 3 * 32}] & /@ Z[1], W[2]];
Z[2] = Ramp[A[2]];
A[3] = Dot[ArrayReshape[Partition[#, {3, 3}, {1, 1}], {22, 22, 3 * 3 * 64}] & /@ Z[2], W[3]];
Z[3] = Ramp[A[3]];
A[4] = Transpose[maxpool[Z[3], {1, 2, 2, 1}], {1, 3, 4, 2}];
Z[4] = Flatten /@ A[4];
A[5] = # + b[5] & /@ (Z[4] . Transpose[W[5]]);
Z[5] = Ramp[A[5]];
A[6] = # + b[6] & /@ (Z[5] . Transpose[W[6]]);
Z[6] = Ramp[A[6]];
A[7] = # + b[7] & /@ (Z[6] . Transpose[W[7]]);
y = Exp[A[7]];
y = y / (Total /@ y);
labelMat = UnitVector[nClasses, #] & /@ rVec;
e[7] = y - labelMat;
Do[e[j] = e[j + 1] . W[j + 1] * Unitize[Z[j]], {j, 6, 5, -1}];
e[4] = e[5] . W[5];
e[3] = (Transpose[maxpoolD[Z[3], {1, 2, 2, 1}] * Unitize[Z[3]], {1, 3, 4, 2}]) * (KroneckerProduct[#, ConstantArray[1., {2, 2}]] & /@ ArrayReshape[#, {64, 11, 11}] & /@ e[4]);
e[2] = Transpose[Map[Flatten, Partition[#, {3, 3}, {1, 1}] & /@ Transpose[ArrayPad[e[3], {{0, 0}, {0, 0}, {2, 2}, {2, 2}}], {1, 4, 2, 3}], {3}] . constructedW[3] * Unitize[Z[2]], {1, 3, 4, 2}];
e[1] = Transpose[Map[Flatten, Partition[#, {3, 3}, {1, 1}] & /@ Transpose[ArrayPad[e[2], {{0, 0}, {0, 0}, {2, 2}, {2, 2}}], {1, 4, 2, 3}], {3}] . constructedW[2] * Unitize[Z[1]], {1, 3, 4, 2}];
Do[gradQW[j] = Transpose[e[j]] . Z[j - 1]; gradQb[j] = Total[e[j]], {j, 5, 7}];
g3 = arrayContract[ArrayReshape[Partition[#, {22, 22}, {1, 1}], {3, 3, 22 * 22, 64}] & /@ Z[2], {1, 4}, ArrayReshape[#, {64, 22 * 22}] & /@ e[3], {1, 3}];
g2 = arrayContract[ArrayReshape[Partition[#, {24, 24}, {1, 1}], {3, 3, 24 * 24, 32}] & /@ Z[1], {1, 4}, ArrayReshape[#, {64, 24 * 24}] & /@ e[2], {1, 3}];
g1 = arrayContract[ArrayReshape[Partition[#, {26, 26}, {1, 1}], {3, 3, 26 * 26}] & /@ Z[0], {1, 4}, ArrayReshape[#, {32, 26 * 26}] & /@ e[1], {1, 3}];
gradQW[3] = Transpose[ArrayReshape[g3, {64, 576}]];
gradQW[2] = Transpose[ArrayReshape[g2, {64, 288}]];
gradQW[1] = Transpose[ArrayReshape[g1, {32, 9}]]];
If[epoch > 10, lr = 10^-3];
If[epoch > 25, lr = 10^-4];
Do[W[index] -= lr * gradQW[index] / nBatch, {index, Windices}];
Do[b[index] -= lr * gradQb[index] / nBatch, {index, bindices}], {p, nBatches}];
preds = batchPred[data];
matches[epoch] = MapThread[Equal, {preds, labels}];
correct[epoch] = Count[matches[epoch], True];
acc[epoch] = N[correct[epoch] / nTrain];
preds = batchPred[testData];
testMatches[epoch] = MapThread[Equal, {preds, testLabels}];
testCorrect[epoch] = Count[testMatches[epoch], True];
testAcc[epoch] = N[testCorrect[epoch] / nTest];
Print[{correct[epoch], acc[epoch], testCorrect[epoch], testAcc[epoch]}], {epoch, epochs}], {epoch, epochs, p, nBatches}]
