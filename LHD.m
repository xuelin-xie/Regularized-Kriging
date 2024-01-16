%�����XL,XU,Ϊ�����������ص�S��ÿ����������һ�����ݵ�
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