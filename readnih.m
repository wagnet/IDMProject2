function [A]= readnih(filename,maxlines)
%% Read in one of the NIH sparse data files 
% assumes there are 6253 features possible.
% Turns each line into a nonsparse line. 
% maxline can be used to limit the total number of lines read
% Warning feature numbering was done c style so we need to add 1
% to get rid of 0
maxfeat=6254;
fid = fopen(filename);
y = 0;
IDNUM =0;
A=[];
A=sparse(A);
tline = fgetl(fid);
while (ischar(tline)&& y<maxlines)
   b=sscanf(tline,'%d ');
   y=y+1;
   temp=zeros(1,maxfeat);
   temp(1,b+1)=1;
   tline = fgetl(fid);
   A=[A ; temp];
end
fclose(fid);