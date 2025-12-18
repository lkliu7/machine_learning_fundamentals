mnist=ResourceData["MNIST","TrainingData"];
mnistTest=ResourceData["MNIST","TestData"];
group[i_]:=group[i]=Select[mnist,#[[2]]==i&][[All,1]]
trainSize[i_]:=trainSize[i]=Length[group[i]]
groupTest[i_]:=groupTest[i]=Select[mnistTest,#[[2]]==i&][[All,1]]
testSize[i_]:=testSize[i]=Length[groupTest[i]]
nums={5,8};
data=Join@@group/@nums;
n=Total[trainSize/@nums];
vec[img_]:=1-Flatten[ImageData[img]]
X=vec/@data;
y=Join[ConstantArray[1,Length[group[nums[[1]]]]],ConstantArray[-1,Length[group[nums[[2]]]]]];
shift=10^-2;
wExact=Inverse[Transpose[X].X+shift IdentityMatrix[784]].Transpose[X].y;
yX=y.X;
XTX=Transpose[X].X;
error[w_]:=Norm[X.w-y]^2
grad[w_]:=2(XTX.w-yX)
error[wExact]
lr=1.5 10^-6;
iters=100000;
w=ConstantArray[0,784];
tol=10^-6;

(* :!CodeAnalysis::BeginBlock:: *)
(* :!CodeAnalysis::Disable::SuspiciousSessionSymbol:: *)

Print[0," ",prevError=error[w]]
AbsoluteTiming[Do[w-=lr grad[w];
currentError=error[w];
If[Mod[i,100]==0,Print[i," ",error[w]]];
If[Abs[prevError/currentError-1]<tol,Print["converged at iteration ", i];Break[]];
prevError=currentError,{i,iters}]]
batchSize=64;
batchGrad[w_,vecs_,labels_]:=2(vecs.w.vecs-labels.vecs)
batchSGD[batchSize_,lr_]:=Block[{batches=Partition[RandomSample[Range[n]],batchSize]},
Do[wSGD-=lr batchGrad[wSGD,X[[batch]],y[[batch]]],{batch,batches}]]
lr=1.5 10^-6;
iters=100000;
wSGD=ConstantArray[0,784];
tol=10^-6;
Print[0," ",prevError=error[wSGD]]
AbsoluteTiming[Do[batchSGD[batchSize,lr];
currentError=error[wSGD];
If[Mod[i,100]==0,Print[i," ",error[wSGD]]];
If[Abs[prevError/currentError-1]<tol,Print["converged at iteration ", i];Break[]];
prevError=currentError,{i,iters}]]
pred[w_,x_]:=Sign[w.x]
predData[w_,data_]:=Sign[data.w]
Print[0," ",prevError=error[w]]
AbsoluteTiming[Do[w-=lr grad[w];
currentError=error[w];
If[Mod[i,100]==0,Print[i," ",error[w]]];
If[Abs[prevError/currentError-1]<tol,Print["converged at iteration ", i];Break[]];
prevError=currentError,{i,iters}]]
pred[w_,x_]:=Sign[w.x]
predData[w_,data_]:=Sign[data.w]
trainAcc1=N[Count[predData[w,X[[;;trainSize[nums[[1]]]]]],1]/trainSize[nums[[1]]]]
trainAcc2=N[Count[predData[w,X[[-trainSize[nums[[2]]];;]]],-1]/trainSize[nums[[2]]]]
testAcc1=N[Count[predData[w,(vec/@groupTest[nums[[1]]])],1]/testSize[nums[[1]]]]
testAcc2=N[Count[predData[w,(vec/@groupTest[nums[[2]]])],-1]/testSize[nums[[2]]]]
trainAccSGD1=N[Count[predData[wSGD,X[[;;trainSize[nums[[1]]]]]],1]/trainSize[nums[[1]]]]
trainAccSGD2=N[Count[predData[wSGD,X[[-trainSize[nums[[2]]];;]]],-1]/trainSize[nums[[2]]]]
testAccSGD1=N[Count[predData[wSGD,(vec/@groupTest[nums[[1]]])],1]/testSize[nums[[1]]]]
testAccSGD2=N[Count[predData[wSGD,(vec/@groupTest[nums[[2]]])],-1]/testSize[nums[[2]]]]
trainAccExact1=N[Count[predData[wExact,X[[;;trainSize[nums[[1]]]]]],1]/trainSize[nums[[1]]]]
trainAccExact2=N[Count[predData[wExact,X[[-trainSize[nums[[2]]];;]]],-1]/trainSize[nums[[2]]]]
testAccExact1=N[Count[predData[wExact,(vec/@groupTest[nums[[1]]])],1]/testSize[nums[[1]]]]
testAccExact2=N[Count[predData[wExact,(vec/@groupTest[nums[[2]]])],-1]/testSize[nums[[2]]]]
Print[{trainAcc1,trainAcc2,testAcc1,testAcc2}]
Print[{trainAccSGD1,trainAccSGD2,testAccSGD1,testAccSGD2}]
Print[{trainAccExact1,trainAccExact2,testAccExact1,testAccExact2}]

(* :!CodeAnalysis::EndBlock:: *)