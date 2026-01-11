#!/usr/bin/env wolframscript
(* ::Package:: *)

dims={20,50,100};


enwik8=Import["machine_learning_fundamentals/enwik9","Text"];
(* remove HTML and XML tags *)
enwik8=StringReplace[enwik8,{"<"~~Shortest[___]~~">"->" ","&"~~Shortest[___]~~";"->" ","#"~~Shortest[___]~~";"->" ","http"~~Shortest[___]~~" "->" "}];
enwik8=RemoveDiacritics[enwik8];
(* keep only alphanumeric characters and spaces *)
enwik8=StringReplace[enwik8,Except[WordCharacter|WhitespaceCharacter]..->""];
(* lowercase all text *)
enwik8=ToLowerCase[enwik8];
(* list of words *)
words=StringSplit[enwik8];
wordCounts=Counts[words];
mostCommonWordsCounts=TakeLargest[wordCounts,10000];
mostCommonWords=Keys[mostCommonWordsCounts];
wordSimPairs=Rest[Import["machine_learning_fundamentals/wordsim353crowd.csv"]];
wordSimWords=ToLowerCase[DeleteDuplicates[Flatten[Most/@wordSimPairs]]];
mostCommonWords=Join[mostCommonWords,Complement[wordSimWords,mostCommonWords]];
wordIndex=AssociationThread[mostCommonWords,Range[Length[mostCommonWords]]];
stream=StringToStream[enwik8];
nCols=Length[mostCommonWords];
row=1;
AbsoluteTiming[Monitor[{rowInds,colInds,vals}=Reap[While[StringQ[line=ReadLine[stream]],w=StringSplit[line];
If[Length[w]<=3,Continue[]];
ids=Lookup[wordIndex,w,Nothing];
cnts=Tally[ids];
cols=cnts[[All,1]];
cts=cnts[[All,2]];
Sow[ConstantArray[row,Length[cols]],"r"];
Sow[cols,"c"];
Sow[cts,"v"];
row++],_,Developer`ToPackedArray[Flatten[#2]]&][[2]];,row]]
Close[stream];
documentWordFrequencyMatrix=SparseArray[Transpose[{rowInds,colInds}]->vals,{row-1,nCols}];


matrixFactorization[mat_,k_,l1_,l2_,steps_]:=Block[{m,n,A,B,U,V},
{n,m}=Dimensions[mat];
V=RandomReal[{-1,1},{m,k}];
Do[A=Inverse[Transpose[V] . V+l1 IdentityMatrix[k]];
U=V . Transpose[A];
U=mat . U;
B=Inverse[Transpose[U] . U+l2 IdentityMatrix[k]];
V=B . Transpose[U];
V=V . mat;
V=Transpose[V],steps];
{U,V}]
matrixFactorizationSGD[mat_,k_,l1_,l2_,lr_,steps_]:=Block[{pos,U,V,u,v,du,dv,t,i,j,n,m},
{n,m}=Dimensions[mat];
V=RandomReal[{-1,1},{m,k}];
U=RandomReal[{-1,1},{n,k}];
Do[{i,j}=1+RandomInteger/@{n-1,m-1};
u=U[[i]];
v=V[[j]];
t=-2(mat[[i,j]]-u . v);
du=t v+2l1 u;
dv=t u+2l2 v;
u-=lr du;
v-=lr dv;
U[[i]]=u;
V[[j]]=v,{p,steps}];
{U,V}]


AbsoluteTiming[MaxMemoryUsed[{Usvd[#],Ssvd[#],Vsvd[#]}=SingularValueDecomposition[N[documentWordFrequencyMatrix],#]]]&/@dims
AbsoluteTiming[MaxMemoryUsed[{Ualt[#],Valt[#]}=matrixFactorization[documentWordFrequencyMatrix,#,1,1,10]]]&/@dims
AbsoluteTiming[MaxMemoryUsed[{Usgd[#],Vsgd[#]}=matrixFactorizationSGD[documentWordFrequencyMatrix,#,1,1,10^-2,10^6]]]&/@dims


(* N.B.:The following reconstruction error computation is highly memory-intensive. Uncomment it if you are sure you want to run it. *)
(*sparseReconstructionError[mat_,U_,S_,V_]:=Block[{reconstructed=U.S.Transpose[V],norm1,norm2,tr},
norm1 = Norm[mat,"Frobenius"]^2;
norm2=Total[reconstructed^2,2];
tr=2Tr[Transpose[mat].reconstructed];
Sqrt[norm1+norm2-tr]]
{sparseReconstructionError[documentWordFrequencyMatrix,Ualt[#],IdentityMatrix[#],Valt[#]],sparseReconstructionError[documentWordFrequencyMatrix,Usgd[#],IdentityMatrix[#],Vsgd[#]],sparseReconstructionError[documentWordFrequencyMatrix,Usvd[#],Ssvd[#],Vsvd[#]]}&/@dims*)


methods={"alt","sgd","svd"};
Do[cosineVsHuman=Reap[Do[words=ToLowerCase[Most[pair]];
positions=wordIndex/@words;
vecs=ToExpression["V"<>method][dim][[positions]];
Sow[{vecs[[1]] . vecs[[2]]/(Norm[vecs[[1]]]Norm[vecs[[2]]]),pair[[-1]]}],{pair,wordSimPairs}]][[2,1]];
corr[method,dim]=Correlation@@Transpose[cosineVsHuman];
listplot[method,dim]=ListPlot[cosineVsHuman,PlotLabel->method<>" method, dim = "<>ToString[dim]<>", correlation = "<>ToString[corr[method,dim]],AxesLabel->{"cosine similarity","human similarity"}],{dim,dims},{method,methods}]


GraphicsGrid[Table[listplot[method,dim],{method,methods},{dim,dims}],ImageSize->Full]
