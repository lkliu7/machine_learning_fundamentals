#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Section:: *)
(*Setup*)


mnist=ResourceData["MNIST","TrainingData"];
mnistTest=ResourceData["MNIST","TestData"];
group[i_]:=group[i]=Select[mnist,#[[2]]==i&][[All,1]]
trainSize[i_]:=trainSize[i]=Length[group[i]]
groupTest[i_]:=groupTest[i]=Select[mnistTest,#[[2]]==i&][[All,1]]
testSize[i_]:=testSize[i]=Length[groupTest[i]]
vec[img_]:=1-Flatten[ImageData[img]]


(* ::Section:: *)
(*Primary formulation*)


h[x_]:=Ramp[1-x]
th[x_]:=Boole[x>=0]
error[w_,b_,c_,X_,y_]:=Norm[w]^2/2+c Total[h[y(X . w+b)]]
gradw[w_,b_,c_,X_,y_]:=Block[{p=1-y(X . w+b)},w-c Total[X y th/@p]]
gradb[w_,b_,c_,X_,y_]:=Block[{p=1-y(X . w+b)},-c Total[y th/@p]]


nums={5,8};
data=Join@@group/@nums;
X=vec/@data;
y=Join[ConstantArray[1,Length[group[nums[[1]]]]],ConstantArray[-1,Length[group[nums[[2]]]]]];
n=Length[y];


lr=1 10^-4;
iters=100000;
w=ConstantArray[0,784];
b=0;
c=0.25;
tol=10^-6;
Print[0," ",prevError=error[w,b,c,X,y]]
AbsoluteTiming[Do[w-=lr gradw[w,b,c,X,y];
b-=lr gradb[w,b,c,X,y];
currentError=error[w,b,c,X,y];
If[Mod[i,100]==0,Print[i," ",currentError]];
If[Abs[prevError/currentError-1]<tol,Print["converged at iteration ", i];Break[]];
prevError=currentError,{i,iters}]]


(* ::Section:: *)
(*Multi-class SVM*)


h[x_]:=Ramp[1-x]
th[x_]:=Boole[x>=0]
error[w_,b_,c_,X_,y_]:=Norm[w]^2/2+c Total[h[y(X . w+b)]]
gradw[w_,b_,c_,X_,y_]:=Block[{p=1-y(X . w+b)},w-c Total[X y th/@p]]
gradb[w_,b_,c_,X_,y_]:=Block[{p=1-y(X . w+b)},-c Total[y th/@p]]
pairs=Subsets[Range[0,9],{2}];
lr=1 10^-4;
iters=100000;
c=0.25;
tol=10^-6;
results=ParallelTable[Block[{data,X,y,n,w,b,prevError,currentError},
data=Join@@group/@nums;
X=vec/@data;
y=Join[ConstantArray[1,Length[group[nums[[1]]]]],ConstantArray[-1,Length[group[nums[[2]]]]]];
n=Length[y];
w=ConstantArray[0,784];
b=0;
prevError=error[w,b,c,X,y];
Do[w-=lr gradw[w,b,c,X,y];
b-=lr gradb[w,b,c,X,y];
currentError=error[w,b,c,X,y];
If[Abs[prevError/currentError-1]<tol,Break[]];
prevError=currentError,{i,iters}];
{nums,w,b}],{nums,pairs},Method->"FinestGrained"];
Do[w[result[[1]]]=result[[2]];b[result[[1]]]=result[[3]],{result,results}]
predPair[pair_,img_]:=Block[{vec=vec[img]},pair[[-Sign[w[pair] . vec+b[pair]]/2+3/2]]]
pred[img_]:=RandomChoice[MaximalBy[Tally[Table[predPair[pair,img],{pair,pairs}]],Last]][[1]]


ParallelTable[N[Counts[pred/@group[i]]/trainSize[i]],{i,0,9}]


(* ::Section:: *)
(*Dual formulation*)


nums={5,8};
data=Join@@group/@nums;
X=vec/@data;
y=Join[ConstantArray[1,Length[group[nums[[1]]]]],ConstantArray[-1,Length[group[nums[[2]]]]]];
n=Length[y];
yyT=Outer[Times,y,y];
Q=yyT (X . Transpose[X]);


error[\[Alpha]_]:=\[Alpha] . Q . \[Alpha]/2-Total[\[Alpha]]
grad[\[Alpha]_]:=Q . \[Alpha]-1
proj[vec_]:=vec-y . vec y/n
projC=Compile[{{vec,_Real,1}},vec-y . vec y/n];


lr=4 10^-5;
iters=1000000;
\[Alpha]=ConstantArray[0,n];
tol=10^-6;
clip=0.5;
Print[0," ",prevError=error[\[Alpha]]]
AbsoluteTiming[Do[\[Alpha]-=lr grad[\[Alpha]];
While[Abs[y . \[Alpha]]>tol,\[Alpha]=Clip[projC[\[Alpha]],{0,clip}]];
currentError=error[\[Alpha]];
If[Mod[i,100]==0,Print[i," ",currentError," ",Max[\[Alpha]]," ",y . \[Alpha]]];
If[Abs[prevError/currentError-1]<tol,Print["converged at iteration ", i];Break[]];
If[currentError>0,Break[]];
prevError=currentError,{i,iters}]]


MatrixPlot[{\[Alpha]}]


w=(\[Alpha] y) . X;
pos=FirstPosition[\[Alpha],Max[\[Alpha]]][[1]];
b=y[[pos]]-w . X[[pos]];


(* ::Section:: *)
(*Results*)


pred[w_,x_,b_]:=Sign[w . x+b]
predData[w_,data_,b_]:=Sign[data . w+b]
N[Count[predData[w,X[[;;trainSize[nums[[1]]]]],b],1]/trainSize[nums[[1]]]]
N[Count[predData[w,X[[-trainSize[nums[[2]]];;]],b],-1]/trainSize[nums[[2]]]]
N[Count[predData[w,(vec/@groupTest[nums[[1]]]),b],1]/testSize[nums[[1]]]]
N[Count[predData[w,(vec/@groupTest[nums[[2]]]),b],-1]/testSize[nums[[2]]]]
