function [y] = langermann(xx, m, c, A)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% LANGERMANN FUNCTION
%
% Authors: Sonja Surjanovic, Simon Fraser University
%          Derek Bingham, Simon Fraser University
% Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
%
% Copyright 2013. Derek Bingham, Simon Fraser University.
%
% THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
% FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
% derivative works, such modified software should be clearly marked.
% Additionally, this program is free software; you can redistribute it 
% and/or modify it under the terms of the GNU General Public License as 
% published by the Free Software Foundation; version 2.0 of the License. 
% Accordingly, this program is distributed in the hope that it will be 
% useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
% of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
% General Public License for more details.
%
% For function details and reference information, see:
% 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INPUTS:
%
% xx = [x1, x2, ..., xd]
% m  = constant (optional), with default value 5
% c  = m-dimensional vector (optional), with default value [1, 2, 5, 2, 3]
%      (when m=5)
% A  = (mxd)-dimensional matrix (optional), with default value
%      [3, 5; 5, 2; 2, 1; 1, 4; 7, 9] (when m=5 and d=2)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d = length(xx);

if (nargin < 2)
    m = 5;
end

if (nargin < 3)
    if (m == 5)
        c = [1, 2, 5, 2, 3];
    else
        error('Value of the m-dimensional vector c is required.')
    end
end

if (nargin < 4)
    if (m==5 && d==2)
        A = [3, 5; 5, 2; 2, 1; 1, 4; 7, 9];
    else
        error('Value of the (mxd)-dimensional matrix A is required.')
    end
end

outer = 0;
for ii = 1:m
    inner = 0;
    for jj = 1:d
        xj = xx(jj);
        Aij = A(ii,jj);
        inner = inner + (xj-Aij)^2;
    end
    new = c(ii) * exp(-inner/pi) * cos(pi*inner);
    outer = outer + new;
end

y = outer;

end
