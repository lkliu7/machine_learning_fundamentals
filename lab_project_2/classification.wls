mnist=ResourceData["MNIST","TrainingData"];
mnistTest=ResourceData["MNIST","TestData"];
group[i_]:=group[i]=Select[mnist,#[[2]]==i&][[All,1]];
groupTest[i_]:=groupTest[i]=Select[mnistTest,#[[2]]==i&][[All,1]];
nums={5,8};
data=Join@@group/@nums;
vec[img_]:=Flatten[ImageData[img]]
X=vec/@data;
y=Join[ConstantArray[1,Length[group[nums[[1]]]]],ConstantArray[-1,Length[group[nums[[2]]]]]];
shift=10^-2;
w=Inverse[Transpose[X].X+shift IdentityMatrix[784]].Transpose[X].y;
pred[x_]:=Sign[w.x]
trainAcc1=N[Count[pred/@vec/@group[nums[[1]]],1]/Length[group[nums[[1]]]]]
trainAcc2=N[Count[pred/@vec/@group[nums[[2]]],-1]/Length[group[nums[[2]]]]]
testAcc1=N[Count[pred/@vec/@groupTest[nums[[1]]],1]/Length[groupTest[nums[[1]]]]]
testAcc2=N[Count[pred/@vec/@groupTest[nums[[2]]],-1]/Length[groupTest[nums[[2]]]]]
Print[{trainAcc1,trainAcc2,testAcc1,testAcc2}]