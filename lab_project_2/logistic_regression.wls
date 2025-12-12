mnist=ResourceData["MNIST","TrainingData"];
mnistTest=ResourceData["MNIST","TestData"];
group[i_]:=group[i]=Select[mnist,#[[2]]==i&][[All,1]]
trainSize[i_]:=trainSize[i]=Length[group[i]]
groupTest[i_]:=groupTest[i]=Select[mnistTest,#[[2]]==i&][[All,1]]
testSize[i_]:=testSize[i]=Length[groupTest[i]]
nums={5,8};
data=Join@@group/@nums;
vec[img_]:=1-Flatten[ImageData[img]]
oneminuslC=Compile[{{x,_Real,1}},1-1/(1+E^-x)];
llC=Compile[{{x,_Real,1}},Log[1/(1+E^-x)]];
X=vec/@data;
y=Join[ConstantArray[1,Length[group[nums[[1]]]]],ConstantArray[-1,Length[group[nums[[2]]]]]];
error[w_]:=-Total[llC[y X.w]]
grad[w_]:=Block[{p=oneminuslC[y X.w]},-(y p).X]
lr=1 10^-4;
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
pred[x_]:=Sign[w.x]
predData[data_]:=Sign[data.w]
trainAcc1=N[Count[predData[w,X[[;;trainSize[nums[[1]]]]]],1]/trainSize[nums[[1]]]]
trainAcc2=N[Count[predData[w,X[[-trainSize[nums[[2]]];;]]],-1]/trainSize[nums[[2]]]]
testAcc1=N[Count[predData[w,(vec/@groupTest[nums[[1]]])],1]/testSize[nums[[1]]]]
testAcc2=N[Count[predData[w,(vec/@groupTest[nums[[2]]])],-1]/testSize[nums[[2]]]]
Print[{trainAcc1,trainAcc2,testAcc1,testAcc2}]

(* :!CodeAnalysis::EndBlock:: *)