mnist=ResourceData["MNIST","TrainingData"];
mnistTest=ResourceData["MNIST","TestData"];
group[i_]:=group[i]=Select[mnist,#[[2]]==i&][[All,1]];
groupTest[i_]:=groupTest[i]=Select[mnistTest,#[[2]]==i&][[All,1]];
nums={5,8};
data=Join@@group/@nums;
vec[img_]:=1-Flatten[ImageData[img]]
l[x_]:=1/(1+E^-x)
lC=Compile[{{x,_Real}},1/(1+E^-x)];
n=Length[data];
X=vec/@data;
y=Join[ConstantArray[1,Length[group[nums[[1]]]]],ConstantArray[-1,Length[group[nums[[2]]]]]];
error[w_]:=Total[l/@(y X.w)]
grad[w_]:=Block[{p=lC/@(y X.w)},(y p(1-p)).X]
lr=10^-3;
iters=1000;
w=ConstantArray[0,784];

(* :!CodeAnalysis::BeginBlock:: *)
(* :!CodeAnalysis::Disable::SuspiciousSessionSymbol:: *)

Print[0," ",error[w]];
Do[w=w-lr grad[w];
If[Mod[i,100]==0,Print[i," ",error[w]]],{i,iters}]
pred[x_]:=Sign[-w.x]
trainAcc1=N[Count[pred/@vec/@group[nums[[1]]],1]/Length[group[nums[[1]]]]]
trainAcc2=N[Count[pred/@vec/@group[nums[[2]]],-1]/Length[group[nums[[2]]]]]
testAcc1=N[Count[pred/@vec/@groupTest[nums[[1]]],1]/Length[groupTest[nums[[1]]]]]
testAcc2=N[Count[pred/@vec/@groupTest[nums[[2]]],-1]/Length[groupTest[nums[[2]]]]]
Print[{trainAcc1,trainAcc2,testAcc1,testAcc2}]

(* :!CodeAnalysis::EndBlock:: *)