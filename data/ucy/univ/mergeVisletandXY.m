%% script that creates an annotation file 6xN
% 3rd and 4th row contains y,x location
% 5th and 6th row contains y',x' vislets 

origFile=csvread('pixel_pos_interpolate.csv')';
visletFile=csvread('vislet_pixel_pos_interpolate_head_added.csv')';



if(size(origFile,1)~=size(visletFile,1))
   error('original annotations and vislet annotation file have different number of annotations')
else
    mergeFile=zeros(size(origFile,1),6);
end
mergeFile(:,1:4)=origFile(:,1:4);
%% copy vislet
mergeFile(:,5:6)=visletFile(:,3:4);
mergeFile=mergeFile';

csvwrite('xy_vislet_pixel_pos.csv',mergeFile)


