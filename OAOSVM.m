function acc = main(file_data)
%trainLabel = file_data1(:,1);
%trainData = file_data1(:,2:end);
%testLabel = file_data2(:,1);
%testData = file_data2(:,2:end);
%[r,c]=size(trainData);
%data=[trainData;testData];
label = file_data(:,1);
data = file_data(:,2:end);

%t_start =cputime;
class=unique(label);
n_class=length(class);
numInst = size(data,1);
%acc=zeros(10,1);
%t_used=zeros(10,1);
%for m=1:10
idx = randperm(numInst);
%idx=1:numInst;
numTrain = fix(0.8*numInst);
numTest = numInst - numTrain;
trainData = data(idx(1:numTrain),:);
testData = data(idx(numTrain+1:end),:);
trainLabel = label(idx(1:numTrain));
testLabel = label(idx(numTrain+1:end));

D=cell(n_class,1);
L=cell(n_class,1);
for i=1:n_class
    c=class(i);
    D{i}=trainData(trainLabel==c,:);
    L{i}=trainLabel(trainLabel==c,1);
end
n=(n_class*(n_class-1))/2;
pred_oao=zeros(numTest,n);
count=0;
%prob = zeros(numTest,n);
for i=1:n_class-1
    [s,~]=size(D{i});
    train(1:s,:)=D{i}(1:s,:);
    %train=D{i};
    group(1:s,1)=1;
    for j=i+1:n_class
        count=count+1;
        [r,~]=size(D{j});
        train(s+1:s+r,:)=D{j}(1:r,:);
        %train=[train;D{j}];
        group(s+1:s+r,1)=-1;
        %options.MaxIter = 1000000;
        %model = svmtrain(train,group,'kernel_function', 'rbf','rbf_sigma',0.1);
        %model = svmtrain(train,group,'Options', options, 'Boxconstraint', 2);
        %model = svmtrain(train,group,'kernel_function', 'polynomial', 'polyorder', 3, 'BoxConstraint', 1000);
        %model = svmtrain(train,group,'Options', options);
        model = svmtrain(train,group,'kernel_function', 'rbf', 'rbf_sigma', 16, 'BoxConstraint', 64);
        %model = svmtrain(train,group,'kernel_function', 'linear', 'BoxConstraint', 1000);
        %index=i(pred==1);
        prob = svmclassify(model, testData);
        %prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
        pred_oao(prob==1,count)=class(i);
        pred_oao(prob==-1,count)=class(j);
     end
end
pred=zeros(numTest,1);
for i=1:numTest
    pred(i,1)=mode(pred_oao(i,:));
end
acc=sum(testLabel==pred)/numel(testLabel);
end
