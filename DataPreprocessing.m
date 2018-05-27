clear;

indexL = {'FOL','FKL','SDL','BSC','STD','LYI'};
TmpL = cell(1,1);


for a = {'FOL','FKL','SDL','BSC'}
for s = [1:23,25:40,42:49,51:67]    
    for t = 1:3        
        if strcmp(a{1},'BSC') == 1 && s == 3 && t == 2
            continue;
        end
        filename = strcat(a{1},'_',num2str(s),'_',num2str(t),'_annotated.csv');
input = readtable(filename);
nR = size(input,1);
Xt= table2array(input(:,3));
Yt = table2array(input(:,4));
Zt = table2array(input(:,5));
Label_t = table2array(input(:,end));
Magnitude_t = sqrt(Xt.^2 + Yt.^2 + Zt.^2);
TA_t = asin(Yt ./ Magnitude_t);
j = 1;
k = 0; 
while k ~= nR   
    k = j + 199;
    if k == 2000
        k = nR;
    end
    if k > nR
        k = nR;
    end
    cnt  = zeros(6,1);
    Tmp = zeros(k-j+1,1);
    Lab = cell(1,1);
    X = Xt(j:k,:);
    Y = Yt(j:k,:);
    Z = Zt(j:k,:);
    Label = Label_t(j:k,:);
    nRp = size(X,1);
    AvgX = mean(X);
    AvgY = mean(Y);
    AvgZ = mean(Z);
    Tmp = AvgX;
    Tmp = horzcat(Tmp, AvgY, AvgZ);
    MedianX = median(X);
    MedianY = median(Y);
    MedianZ = median(Z);    
    Tmp = horzcat(Tmp, MedianX, MedianY, MedianZ);
    StdX = std(X);
    StdY = std(Y);
    StdZ = std(Z);
    Tmp = horzcat(Tmp, StdX, StdY, StdZ);
    SkewX = skewness(X);
    SkewY = skewness(Y);
    SkewZ = skewness(Z);
    Tmp = horzcat(Tmp, SkewX, SkewY, SkewZ);
    KurtosisX = kurtosis(X);
    KurtosisY = kurtosis(Y);
    KurtosisZ = kurtosis(Z);
    Tmp = horzcat(Tmp, KurtosisX, KurtosisY, KurtosisZ);
    MinX = min(X);
    MinY = min(Y);
    MinZ = min(Z);
    Tmp = horzcat(Tmp, MinX, MinY, MinZ);
    MaxX = max(X);
    MaxY = max(Y);
    MaxZ = max(Z);
    Tmp = horzcat(Tmp, MaxX, MaxY, MaxZ);
    Slope = sqrt((MaxX - MinX).^2 + (MaxY - MinY).^2 + (MaxZ - MinZ).^2) ;
    Tmp = horzcat(Tmp, Slope);
    TA = TA_t(j:k,:);
    MeanTA = mean(TA);
    StdTA = std(TA);
    SkewTA = skewness(TA);
    KurtosisTA = kurtosis(TA);
    Tmp = horzcat(Tmp, MeanTA, StdTA, SkewTA, KurtosisTA);
    Abs_X = sum(abs(X - AvgX)) / nRp;
    Abs_Y = sum(abs(Y - AvgY)) / nRp;
    Abs_Z = sum(abs(Z - AvgZ)) / nRp;
    Tmp = horzcat(Tmp, Abs_X, Abs_Y, Abs_Z);   
    Abs_meanX = mean(abs(X));
    Abs_meanY = mean(abs(Y));
    Abs_meanZ = mean(abs(Z));
    Abs_medianX = median(abs(X));
    Abs_medianY = median(abs(Y));
    Abs_medianZ = median(abs(Z));
    Tmp = horzcat(Tmp, Abs_meanX, Abs_meanY, Abs_meanZ, Abs_medianX, Abs_medianY, Abs_medianZ);  
    Abs_stdX = std(abs(X));
    Abs_stdY = std(abs(Y));
    Abs_stdZ = std(abs(Z));
    Abs_skewX = skewness(abs(X));
    Abs_skewY = skewness(abs(Y));
    Abs_skewZ = skewness(abs(Z));
    Tmp = horzcat(Tmp, Abs_stdX, Abs_stdY, Abs_stdZ, Abs_skewX, Abs_skewY, Abs_skewZ);  
    Abs_KurtosisX = kurtosis(abs(X));
    Abs_KurtosisY = kurtosis(abs(Y));
    Abs_KurtosisZ = kurtosis(abs(Z));
    Abs_minX = min(abs(X));
    Abs_minY = min(abs(Y));
    Abs_minZ = min(abs(Z));
    Tmp = horzcat(Tmp, Abs_KurtosisX, Abs_KurtosisY, Abs_KurtosisZ, Abs_minX, Abs_minY, Abs_minZ);  
    Abs_maxX = max(abs(X));
    Abs_maxY = max(abs(Y));
    Abs_maxZ = max(abs(Z));
    Abs_slope = sqrt((Abs_maxX - Abs_minX).^2 + (Abs_maxY - Abs_minY).^2 + (Abs_maxZ - Abs_minZ).^2);
    Tmp = horzcat(Tmp, Abs_maxX, Abs_maxY, Abs_maxZ, Abs_slope); 
    Magnitude = Magnitude_t(j:k,:);
    MeanMag = mean(Magnitude);
    StdMag = std(Magnitude);
    MinMag = min(Magnitude);
    MaxMag = max(Magnitude);
    Tmp = horzcat(Tmp, MeanMag, StdMag, MinMag, MaxMag);
    DiffMinMaxMag = MaxMag - MinMag;
    ZCR_Mag = sum(abs(diff(Magnitude > 0)))/length(Magnitude);
    Tmp = horzcat(Tmp, DiffMinMaxMag, ZCR_Mag);
    AvgResAcc = (1/nRp)*sum(Magnitude);
    Tmp = horzcat(Tmp,AvgResAcc);
    for cL = 1:(k-j+1)
        if strcmp(Label{cL}, 'FOL') == 1 
            cnt(1) = cnt(1) + 1;
        elseif strcmp(Label{cL}, 'FKL') == 1 
            cnt(2) = cnt(2) + 1;
        elseif strcmp(Label{cL}, 'SDL') == 1 
            cnt(3) = cnt(3) + 1;
        elseif strcmp(Label{cL}, 'BSC') == 1 
            cnt(4) = cnt(4) + 1;
        elseif strcmp(Label{cL}, 'STD') == 1 
            cnt(5) = cnt(5) + 1;
        elseif strcmp(Label{cL}, 'LYI') == 1 
            cnt(6) = cnt(6) + 1;          
    end
    end
    
    [MaxL,Index] = max(cnt);
    Lab{1} = indexL{Index};     
        
    if j==1 && s == 1 && t == 1 && strcmp(a{1},'FOL') == 1
        Final = Tmp;
        TmpL = Lab;
    else
        Final = vertcat(Final, Tmp);
        TmpL = vertcat(TmpL, Lab);
    end
    j = k + 1;
end

    end
end
end

TmpLW = array2table(TmpL);
FinalW = array2table(Final);

%%
FinalW = [FinalW TmpLW];
%%
FinalW.Properties.VariableNames = {'AvgX','AvgY','AvgZ','MedianX','MedianY','MedianZ','StdX',...
    'StdY','StdZ','SkewX','SkewY','SkewZ','KurtosisX','KurtosisY','KurtosisZ','MinX','MinY',...
    'MinZ','MaxX','MaxY','MaxZ','Slope','MeanTA','StdTA','SkewTA','KurtosisTA',...
    'AbsX','AbsY','AbsZ','AbsMeanX','AbsMeanY','AbsMeanZ','AbsMedianX','AbsMedianY','AbsMedianZ',...
    'AbsStdX','AbsStdY','AbsStdZ','AbsSkewX','AbsSkewY','AbsSkewZ',...
    'AbsKurtosisX','AbsKurtosisY','AbsKurtosisZ','AbsMinX','AbsMinY','AbsMinZ',...
    'AbsMaxX','AbsMaxY','AbsMaxZ','AbsSlope','MeanMag',...
    'StdMag','MinMag','MaxMag','DiffMinMaxMag','ZCR_Mag','AverageResultantAcceleration','Label'};
writetable(FinalW,'FOL_FKL_BSC_SDL_STD_LYI.csv','Delimiter',',');
