mnist = ResourceData["MNIST", "TrainingData"];
mnistTest = ResourceData["MNIST", "TestData"];
group[i_] := group[i] = Select[mnist, #[[2]] == i &][[All, 1]]
trainSize[i_] := trainSize[i] = Length[group[i]]
groupTest[i_] := groupTest[i] = Select[mnistTest, #[[2]] == i &][[All, 1]]
testSize[i_] := testSize[i] = Length[groupTest[i]]
nums = {5, 8};
data = Join @@ group /@ nums;
n = Total[trainSize /@ nums];
vec[img_] := 1 - Flatten[ImageData[img]]
l[x_] := 1/(1 + E^-x)
lC = Compile[{{x, _Real, 1}}, 1/(1 + E^-x)];
X = vec /@ data;
y = Join[ConstantArray[1, Length[group[nums[[1]]]]], ConstantArray[-1, Length[group[nums[[2]]]]]];
error[w_] := Total[lC[y X . w]]
grad[w_] := Block[{p = lC[y X . w]}, (y p (1 - p)) . X]
lr = 10^-3;
iters = 100000;
w = ConstantArray[0, 784];
tol = 10^-6;

(* :!CodeAnalysis::BeginBlock:: *)
(* :!CodeAnalysis::Disable::SuspiciousSessionSymbol:: *)

Print[0, " ", prevError = error[w]]
AbsoluteTiming[Do[w -= lr grad[w];
currentError = error[w];
If[Mod[i, 100] == 0, Print[i, " ", currentError]];
If[Abs[prevError/currentError - 1] < tol, Print["converged at iteration ", i]; Break[]];
prevError = currentError, {i, iters}]]
batchSize = 64;
grad[w_] := Block[{p = lC[y X . w]}, (y p (1 - p)) . X]
batchGrad[w_, vecs_, labels_] := Block[{p = lC[labels vecs . w]}, (labels p (1 - p)) . vecs]
batchSGD[batchSize_, lr_] := Block[{batches = Partition[RandomSample[Range[n]], batchSize]},
Do[wSGD -= lr batchGrad[wSGD, X[[batch]], y[[batch]]], {batch, batches}]]
lr = 10^-3;
iters = 100000;
wSGD = ConstantArray[0, 784];
tol = 10^-6;
Print[0," ",prevError=error[wSGD]]
AbsoluteTiming[Do[batchSGD[batchSize, lr];
currentError=error[wSGD];
If[Mod[i, 100] == 0, Print[i, " ", currentError]];
If[Abs[prevError/currentError - 1] < tol, Print["converged at iteration ", i]; Break[]];
prevError = currentError, {i, iters}]]
pred[w_, x_] := Sign[-w . x]
predData[w_, data_] := Sign[-data . w]
trainAcc1 = N[Count[predData[w, X[[;; trainSize[nums[[1]]]]]], 1]/trainSize[nums[[1]]]]
trainAcc2 = N[Count[predData[w, X[[-trainSize[nums[[2]]];; ]]], -1]/trainSize[nums[[2]]]]
testAcc1 = N[Count[predData[w, (vec /@ groupTest[nums[[1]]])], 1]/testSize[nums[[1]]]]
testAcc2 = N[Count[predData[w, (vec /@ groupTest[nums[[2]]])], -1]/testSize[nums[[2]]]]
trainAccSGD1 = N[Count[predData[wSGD, X[[;; trainSize[nums[[1]]]]]], 1]/trainSize[nums[[1]]]]
trainAccSGD2 = N[Count[predData[wSGD, X[[-trainSize[nums[[2]]];; ]]], -1]/trainSize[nums[[2]]]]
testAccSGD1 = N[Count[predData[wSGD, (vec /@ groupTest[nums[[1]]])], 1]/testSize[nums[[1]]]]
testAccSGD2 = N[Count[predData[wSGD, (vec /@ groupTest[nums[[2]]])], -1]/testSize[nums[[2]]]]
Print["MCE loss - Full GD: ", error[w], ", SGD: ", error[wSGD]]
Print["Full GD accuracies - Train digit ", nums[[1]], ": ", trainAcc1, ", Train digit ", nums[[2]], ": ", trainAcc2, ", Test digit ", nums[[1]], ": ", testAcc1, ", Test digit ", nums[[2]], ": ", testAcc2]
Print["SGD accuracies - Train digit ", nums[[1]], ": ", trainAccSGD1, ", Train digit ", nums[[2]], ": ", trainAccSGD2, ", Test digit ", nums[[1]], ": ", testAccSGD1, ", Test digit ", nums[[2]], ": ", testAccSGD2]

(* :!CodeAnalysis::EndBlock:: *)