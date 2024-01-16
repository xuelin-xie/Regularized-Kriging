%输入的XL,XU,为列向量，返回的S中每个行向量是一个数据点
%%
function S = LHD(XL, XU, numPts)
    p = length(XL);
    S0 = lhsamp(numPts, p);
    delta = XU - XL;
    S = [];
    for k=1:numPts
        Sk = XL + delta.*S0(k,:);
        S = [S; Sk];
    end
return
%%